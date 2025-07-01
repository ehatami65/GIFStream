import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform


@dataclass
class Compression:
    """Uses quantization and sorting to compress splats into WebP files and uses
    K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `Pillow <https://pypi.org/project/pillow/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
        lossless (bool, optional): Whether to use lossless WebP compression. Defaults to True.
        quality (int, optional): Quality of WebP compression (0-100). Only used when lossless is False. Defaults to 100.
        seed (int, optional): Seed for sorting. Defaults to None.
    """

    use_sort: bool = True
    verbose: bool = True
    lossless: bool = True
    quality: int = 100
    seed: int = None

    def _get_compress_fn(self, param_name: str) -> Callable:
        compress_fn_map = {
            "means": _compress_webp_16bit,
            "scales": _compress_webp,
            "quats": _compress_webp,
            "opacities": _compress_webp,
            "sh0": _compress_webp,
            "shN": _compress_kmeans,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        decompress_fn_map = {
            "means": _decompress_webp_16bit,
            "scales": _decompress_webp,
            "quats": _decompress_webp,
            "opacities": _decompress_webp,
            "sh0": _decompress_webp,
            "shN": _decompress_kmeans,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(
        self,
        splats: Dict[str, Tensor],
        quality_settings: Optional[Dict[str, Any]] = None,
        sort_indices: Optional[Tensor] = None,
        force_resort: bool = False,
        shn_initial_centroids: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Tensor], Optional[Tensor]]:
        """Run compression

        Args:
            splats (Dict[str, Tensor]): Gaussian splats to compress
            quality_settings (Dict[str, Any], optional): Per-parameter quality settings.
                E.g. {"means": {"lossless": False, "quality": 80}}. Defaults to class defaults.
            sort_indices (Tensor, optional): Pre-computed sorting indices to apply. If None,
                new indices will be computed. Defaults to None.
            force_resort (bool, optional): If True, re-sorts the splats even if sort_indices
                are provided, using them as an initialization. Defaults to False.
            shn_initial_centroids (Tensor, optional): Pre-computed centroids for SH K-means.
                If None, new centroids will be computed. Defaults to None.
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Tensor], Optional[Tensor]]: 
                - Metadata dictionary
                - Dictionary of parameter names to numpy arrays
                - Computed sort indices (if they were computed in this call)
                - Computed SH centroids (if they were computed in this call)
        """
        # Work on a deep copy to avoid modifying the original splats
        splats = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in splats.items()}

        # Param-specific preprocessing
        if "means" in splats and splats["means"] is not None:
            splats["means"] = log_transform(splats["means"])
        if "quats" in splats and splats["quats"] is not None:
            splats["quats"] = F.normalize(splats["quats"], dim=-1)

        # Reshape spherical harmonics from (N, D, 3) to (N, D*3) for compression
        if "sh0" in splats and splats["sh0"] is not None and splats["sh0"].ndim == 3:
            splats["sh0"] = splats["sh0"].squeeze(1)  # (N, 1, 3) -> (N, 3)
        if "shN" in splats and splats["shN"] is not None and splats["shN"].ndim == 3:
            shN = splats["shN"]
            if shN.shape[1] > 0:
                splats["shN"] = shN.permute(0, 2, 1).reshape(shN.shape[0], -1)
            else:
                # Handle case with no higher-degree SHs
                splats["shN"] = torch.zeros((shN.shape[0], 0), device=shN.device)

        n_gs = len(splats["means"])
        n_sidelen = int(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2
        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        newly_computed_indices = None
        if self.use_sort:
            if sort_indices is not None and not force_resort:
                # Apply pre-computed indices without sorting
                for k, v in splats.items():
                    splats[k] = v[sort_indices]
            else:
                # Compute new indices, possibly from a warm start
                splats, newly_computed_indices = sort_splats(
                    splats, seed=self.seed, initial_indices=sort_indices
                )

        meta = {}
        compressed_arrays = {}
        new_shn_centroids = None
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)

            # Determine quality settings for this parameter
            param_quality = (quality_settings or {}).get(param_name, {})
            lossless = param_quality.get("lossless", self.lossless)
            quality = param_quality.get("quality", self.quality)

            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
                "lossless": lossless,
                "quality": quality,
                "quality_settings": quality_settings or {},
            }
            if compress_fn is _compress_kmeans:
                kwargs["initial_centroids"] = shn_initial_centroids
                param_meta, param_data, new_shn_centroids = compress_fn(
                    param_name, splats[param_name], **kwargs
                )
            else:
                param_meta, param_data = compress_fn(
                    param_name, splats[param_name], **kwargs
                )

            meta[param_name] = param_meta
            for key, value in param_data.items():
                if key == "arr":  # from npz
                    compressed_arrays[param_name] = value
                elif key == "img":
                    compressed_arrays[param_name] = value
                else:
                    compressed_arrays[f"{param_name}_{key}"] = value

        return meta, compressed_arrays, newly_computed_indices, new_shn_centroids

    def decompress(
        self, meta: Dict[str,Any], compressed_arrays: Dict[str, np.ndarray], device: str = "cpu", to_tensors: bool = True
    ) -> Dict[str, Any]:
        """Run decompression

        Args:
            meta (Dict[str, Any]): metadata dictionary
            compressed_arrays (Dict[str, np.ndarray]): dictionary of param names to numpy arrays
            device (str, optional): device to load tensors to. Defaults to "cpu".
            to_tensors (bool, optional): whether to convert to tensors. Defaults to True.

        Returns:
            Dict[str, Any]: Decompressed Gaussian splats. Either Tensors or Numpy arrays.
        """
        splats = {}

        for param_name, param_meta in meta.items():
            decompress_fn = self._get_decompress_fn(param_name)
            splats[param_name], _ = decompress_fn(
                param_name,
                param_meta,
                compressed_arrays,
                device=device,
                to_tensors=to_tensors,
            )

        # Param-specific postprocessing
        if to_tensors:
            if "means" in splats and splats["means"] is not None:
                splats["means"] = inverse_log_transform(splats["means"])

            num_splats = (
                splats["means"].shape[0]
                if "means" in splats and splats["means"] is not None
                else 0
            )
            
            device = splats["means"].device if "means" in splats and splats["means"] is not None else device

            # Reshape sh0 back to (N, 1, 3)
            if "sh0" in splats and splats["sh0"] is not None and splats["sh0"].ndim == 2:
                splats["sh0"] = splats["sh0"].unsqueeze(1)

            # Reshape shN back to (N, K, 3)
            if "shN" in splats and splats["shN"] is not None:
                shN_flat = splats["shN"]
                if num_splats > 0 and shN_flat.numel() > 0:
                    # The compression flattens shN from (N, K, 3) to (N, 3*K). Reshape it back.
                    K_sh = shN_flat.shape[1] // 3
                    splats["shN"] = shN_flat.reshape(num_splats, 3, K_sh).permute(
                        0, 2, 1
                    )
                else:
                    # Handle case with no higher-degree SHs
                    splats["shN"] = torch.zeros(
                        (num_splats, 0, 3), device=shN_flat.device
                    )
        return splats


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def _precompress_webp(params: Tensor, n_sidelen: int, **kwargs) -> Tuple[Dict, Dict]:
    if torch.numel(params) == 0:
        meta = {"shape": list(params.shape), "dtype": str(params.dtype).split(".")[1]}
        return meta, {}

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    
    grid_norm = (grid - mins) / (maxs - mins)
    
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta, {"img": img}


def _precompress_webp_16bit(params: Tensor, n_sidelen: int, **kwargs) -> Tuple[Dict, Dict]:
    if torch.numel(params) == 0:
        meta = {"shape": list(params.shape), "dtype": str(params.dtype).split(".")[1]}
        return meta, {}

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    
    grid_norm = (grid - mins) / (maxs - mins)

    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)
    
    img_l = (img & 0xFF).astype(np.uint8)
    img_u = ((img >> 8) & 0xFF).astype(np.uint8)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta, {"l": img_l, "u": img_u}


def _compress_webp(
    param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Compress parameters with 8-bit quantization and return as numpy arrays.

    Args:
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - metadata
            - dictionary of image names to numpy arrays
    """
    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta, {}

    meta, images = _precompress_webp(params, n_sidelen, **kwargs)
    return meta, {"img": images["img"]}


def _decompress_webp(
    param_name: str,
    meta: Dict[str, Any],
    image_data: Dict[str, np.ndarray],
    device: str = "cpu",
    to_tensors: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Decompress parameters from numpy array.

    Args:
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata
        image_data (Dict[str, np.ndarray]): numpy array image data
        device (str, optional): device to load to. Defaults to "cpu".
        to_tensors (bool, optional): whether to convert to tensors. Defaults to True.

    Returns:
        Tuple[Any, Dict[str, float]]:
            - Parameters (Tensor or ndarray)
            - Timing dictionary
    """
    io_time = 0.0
    t0_proc = time.time()

    if not np.all(meta["shape"]):
        if to_tensors:
            params = torch.zeros(
                meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device
            )
        else:
            params = np.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        processing_time = time.time() - t0_proc
        return params, {"io_time": io_time, "processing_time": processing_time}

    img = image_data[param_name]

    t0_proc = time.time()
    # Determine the expected number of channels from the metadata
    expected_channels = meta["shape"][-1] if len(meta["shape"]) > 1 else 1

    # If the saved image was grayscale (1 channel), but read as RGB (3 channels), take one channel.
    if img.ndim == 3 and expected_channels == 1:
        img = img[..., 0]
        
    img_norm = img / (2**8 - 1)

    if to_tensors:
        grid_norm = torch.tensor(img_norm, device=device)
        mins = torch.tensor(meta["mins"], device=device)
        maxs = torch.tensor(meta["maxs"], device=device)
        grid = grid_norm * (maxs - mins) + mins
        params = grid.reshape(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
    else:
        mins = np.array(meta["mins"])
        maxs = np.array(meta["maxs"])
        grid = img_norm * (maxs - mins) + mins
        params = grid.reshape(meta["shape"]).astype(meta["dtype"])

    processing_time = time.time() - t0_proc
    timings = {"io_time": io_time, "processing_time": processing_time}
    return params, timings


def _compress_webp_16bit(
    param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Compress parameters with 16-bit quantization and return numpy arrays.

    Args:
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
            - metadata
            - dictionary of image names to numpy arrays
    """
    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta, {}

    meta, images = _precompress_webp_16bit(params, n_sidelen, **kwargs)
    return meta, {"l": images["l"], "u": images["u"]}


def _decompress_webp_16bit(
    param_name: str,
    meta: Dict[str, Any],
    image_data: Dict[str, np.ndarray],
    device: str = "cpu",
    to_tensors: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Decompress parameters from numpy arrays.

    Args:
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata
        image_data (Dict[str, np.ndarray]): numpy array image data
        device (str, optional): device to load to. Defaults to "cpu".
        to_tensors (bool, optional): whether to convert to tensors. Defaults to True.

    Returns:
        Tuple[Any, Dict[str, float]]:
            - Parameters (Tensor or ndarray)
            - Timing dictionary
    """
    io_time = 0.0
    t0_proc = time.time()

    if not np.all(meta["shape"]):
        if to_tensors:
            params = torch.zeros(
                meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device
            )
        else:
            params = np.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        processing_time = time.time() - t0_proc
        return params, {"io_time": io_time, "processing_time": processing_time}

    img_l = image_data[f"{param_name}_l"]
    img_u = image_data[f"{param_name}_u"]

    t0_proc = time.time()
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) | img_l

    img_norm = img / (2**16 - 1)

    if to_tensors:
        grid_norm = torch.tensor(img_norm, device=device)
        mins = torch.tensor(meta["mins"], device=device)
        maxs = torch.tensor(meta["maxs"], device=device)
        grid = grid_norm * (maxs - mins) + mins
        params = grid.reshape(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
    else:
        mins = np.array(meta["mins"])
        maxs = np.array(meta["maxs"])
        grid = img_norm * (maxs - mins) + mins
        params = grid.reshape(meta["shape"]).astype(meta["dtype"])

    processing_time = time.time() - t0_proc
    timings = {"io_time": io_time, "processing_time": processing_time}
    return params, timings


def _compress_npz(
    param_name: str, params: Tensor, **kwargs
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Return parameters as a numpy array."""
    arr = params.detach().cpu().numpy()
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta, {"arr": arr}


def _decompress_npz(
    param_name: str,
    meta: Dict[str, Any],
    array_data: Dict[str, np.ndarray],
    device: str = "cpu",
    to_tensors: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Decompress parameters from numpy array."""
    t0_io = time.time()
    params = array_data[param_name]
    io_time = time.time() - t0_io

    t0_proc = time.time()
    params = params.reshape(meta["shape"])
    if to_tensors:
        params = torch.from_numpy(params).to(device)
        params = params.to(dtype=getattr(torch, meta["dtype"]))
    else:
        params = params.astype(meta["dtype"])

    processing_time = time.time() - t0_proc
    timings = {"io_time": io_time, "processing_time": processing_time}
    return params, timings


def _precompress_kmeans(
    params: Tensor,
    n_sidelen: int,
    quantization: int = 8,
    verbose: bool = True,
    initial_centroids: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Dict, Dict, Tensor]:
    """Runs K-means clustering on parameters and returns centroids and labels as images."""
    try:
        from torchpq.clustering import KMeans
    except ImportError:
        raise ImportError(
            "Please install torchpq with 'pip install torchpq cupy' to use K-means clustering"
        )

    if torch.numel(params) == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta, {}, None

    params = params.reshape(params.shape[0], -1)
    dim = params.shape[1]
    n_clusters = round((len(params) >> 2) / 64) * 64
    n_clusters = min(n_clusters, 2**16)

    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    
    fit_centroids = (
        initial_centroids.permute(1, 0) if initial_centroids is not None else None
    )
    labels = kmeans.fit(params.permute(1, 0).contiguous(), centroids=fit_centroids)
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids)
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)

    sorted_indices = np.lexsort(centroids_quant.T)
    sorted_indices = sorted_indices.reshape(64, -1).T.reshape(-1)
    sorted_centroids_quant = centroids_quant[sorted_indices]
    inverse = np.argsort(sorted_indices)

    centroids_packed = sorted_centroids_quant.reshape(-1, int(dim * 64 / 3), 3)
    labels = inverse[labels].astype(np.uint16).reshape((n_sidelen, n_sidelen))
    labels_l = labels & 0xFF
    labels_u = (labels >> 8) & 0xFF

    labels_combined = np.zeros((n_sidelen, n_sidelen, 3), dtype=np.uint8)
    labels_combined[..., 0] = labels_l.astype(np.uint8)
    labels_combined[..., 1] = labels_u.astype(np.uint8)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
    }
    images = {"centroids": centroids_packed, "labels": labels_combined}
    return meta, images, centroids


def _compress_kmeans(
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    quantization: int = 8,
    verbose: bool = True,
    initial_centroids: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Optional[Tensor]]:
    """Run K-means clustering on parameters and return centroids and labels as numpy arrays.

    .. warning::
        TorchPQ must installed to use K-means clustering.

    Args:
        param_name (str): parameter field name
        params (Tensor): parameters to compress
        n_sidelen (int): image side length
        quantization (int): number of bits in quantization
        verbose (bool, optional): Whether to print verbose information. Default to True.
        initial_centroids (Tensor, optional): Pre-computed centroids for K-means.
            If None, new centroids will be computed. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, np.ndarray]]: 
            - metadata
            - dictionary of image names to numpy arrays
            - The computed centroids
    """
    if torch.numel(params) == 0:
        return (
            {
                "shape": list(params.shape),
                "dtype": str(params.dtype).split(".")[1],
            },
            {},
            None,
        )

    meta, images, new_centroids = _precompress_kmeans(
        params,
        n_sidelen,
        quantization,
        verbose,
        initial_centroids=initial_centroids,
        **kwargs,
    )
    return meta, images, new_centroids


def _decompress_kmeans(
    param_name: str,
    meta: Dict[str, Any],
    image_data: Dict[str, np.ndarray],
    device: str = "cpu",
    to_tensors: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """Decompress parameters from K-means compression.

    Args:
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata
        image_data (Dict[str, np.ndarray]): numpy array image data
        device (str, optional): device to load to. Defaults to "cpu".
        to_tensors: bool, optional): whether to convert to tensors. Defaults to True.

    Returns:
        Tuple[Any, Dict[str, float]]:
            - Parameters (Tensor or ndarray)
            - Timing dictionary
    """
    io_time = 0.0
    t0_proc = time.time()

    if not np.all(meta["shape"]):
        if to_tensors:
            params = torch.zeros(
                meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device
            )
        else:
            params = np.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        processing_time = time.time() - t0_proc
        return params, {"io_time": io_time, "processing_time": processing_time}

    centroids_packed_img = image_data[f"{param_name}_centroids"]
    labels_combined_img = image_data[f"{param_name}_labels"] # Use .get for safety

    t0_proc = time.time()
    # Decompress labels
    labels_l = labels_combined_img[..., 0].astype(np.uint16)
    labels_u = labels_combined_img[..., 1].astype(np.uint16)
    labels = (labels_u << 8) | labels_l

    labels = labels.flatten()

    # Decompress centroids
    dim = np.prod(meta['shape'][1:])
    sorted_centroids_quant = centroids_packed_img.reshape(-1, dim)

    centroids_norm = sorted_centroids_quant / (2 ** meta["quantization"] - 1)
    
    if to_tensors:
        centroids_norm = torch.from_numpy(centroids_norm.astype(np.float32)).to(device)
        mins = torch.tensor(meta["mins"], device=device)
        maxs = torch.tensor(meta["maxs"], device=device)
        centroids = centroids_norm * (maxs - mins) + mins
        params = centroids[labels]
        params = params.reshape(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
    else:
        centroids_norm = centroids_norm.astype(np.float32)
        mins = np.array(meta["mins"])
        maxs = np.array(meta["maxs"])
        centroids = centroids_norm * (maxs - mins) + mins
        params = centroids[labels]
        params = params.reshape(meta["shape"])
        params = params.astype(meta["dtype"])

    processing_time = time.time() - t0_proc
    timings = {"io_time": io_time, "processing_time": processing_time}
    return params, timings
