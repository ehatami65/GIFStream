import math
import struct
from io import BytesIO
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert Sphere Harmonics to RGB

    Args:
        sh (torch.Tensor): SH tensor

    Returns:
        torch.Tensor: RGB tensor
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def part1by2_vec(x: torch.Tensor) -> torch.Tensor:
    """Interleave bits of x with 0s

    Args:
        x (torch.Tensor): Input tensor. Shape (N,)

    Returns:
        torch.Tensor: Output tensor. Shape (N,)
    """

    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8)) & 0x0300F00F
    x = (x ^ (x << 4)) & 0x030C30C3
    x = (x ^ (x << 2)) & 0x09249249
    return x


def encode_morton3_vec(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Compute Morton codes for 3D coordinates

    Args:
        x (torch.Tensor): X coordinates. Shape (N,)
        y (torch.Tensor): Y coordinates. Shape (N,)
        z (torch.Tensor): Z coordinates. Shape (N,)
    Returns:
        torch.Tensor: Morton codes. Shape (N,)
    """
    return (part1by2_vec(z) << 2) + (part1by2_vec(y) << 1) + part1by2_vec(x)


def sort_centers(centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Sort centers based on Morton codes

    Args:
        centers (torch.Tensor): Centers. Shape (N, 3)
        indices (torch.Tensor): Indices. Shape (N,)
    Returns:
        torch.Tensor: Sorted indices. Shape (N,)
    """
    # Compute min and max values in a single operation
    min_vals, _ = torch.min(centers, dim=0)
    max_vals, _ = torch.max(centers, dim=0)

    # Compute the scaling factors
    lengths = max_vals - min_vals
    lengths[lengths == 0] = 1  # Prevent division by zero

    # Normalize and scale to 10-bit integer range (0-1024)
    scaled_centers = ((centers - min_vals) / lengths * 1024).floor().to(torch.int32)

    # Extract x, y, z coordinates
    x, y, z = scaled_centers[:, 0], scaled_centers[:, 1], scaled_centers[:, 2]

    # Compute Morton codes using vectorized operations
    morton = encode_morton3_vec(x, y, z)

    # Sort indices based on Morton codes
    sorted_indices = indices[torch.argsort(morton).to(indices.device)]

    return sorted_indices


def pack_unorm(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a floating point value into an unsigned integer with a given number of bits.

    Args:
        value (torch.Tensor): Floating point value to pack. Shape (N,)
        bits (int): Number of bits to pack into.

    Returns:
        torch.Tensor: Packed value. Shape (N,)
    """

    t = (1 << bits) - 1
    packed = torch.clamp((value * t + 0.5).floor(), min=0, max=t)
    # Convert to integer type
    return packed.to(torch.int64)


def pack_111011(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pack three floating point values into a 32-bit integer with 11, 10, and 11 bits.

    Args:
        x (torch.Tensor): X component. Shape (N,)
        y (torch.Tensor): Y component. Shape (N,)
        z (torch.Tensor): Z component. Shape (N,)
    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """
    # Pack each component using pack_unorm
    packed_x = pack_unorm(x, 11) << 21
    packed_y = pack_unorm(y, 10) << 11
    packed_z = pack_unorm(z, 11)

    # Combine the packed values using bitwise OR
    return packed_x | packed_y | packed_z


def pack_8888(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Pack four floating point values into a 32-bit integer with 8 bits each.

    Args:
        x (torch.Tensor): X component. Shape (N,)
        y (torch.Tensor): Y component. Shape (N,)
        z (torch.Tensor): Z component. Shape (N,)
        w (torch.Tensor): W component. Shape (N,)
    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """
    # Pack each component using pack_unorm
    packed_x = pack_unorm(x, 8) << 24
    packed_y = pack_unorm(y, 8) << 16
    packed_z = pack_unorm(z, 8) << 8
    packed_w = pack_unorm(w, 8)

    # Combine the packed values using bitwise OR
    return packed_x | packed_y | packed_z | packed_w


def pack_rotation(q: torch.Tensor) -> torch.Tensor:
    """Pack a quaternion into a 32-bit integer.

    Args:
        q (torch.Tensor): Quaternions. Shape (N, 4)

    Returns:
        torch.Tensor: Packed values. Shape (N,)
    """

    # Normalize each quaternion
    norms = torch.linalg.norm(q, dim=-1, keepdim=True)
    q = q / norms

    # Find the largest component index for each quaternion
    largest_components = torch.argmax(torch.abs(q), dim=-1)

    # Flip quaternions where the largest component is negative
    batch_indices = torch.arange(q.size(0), device=q.device)
    largest_values = q[batch_indices, largest_components]
    flip_mask = largest_values < 0
    q[flip_mask] *= -1

    # Precomputed indices for the components to pack (excluding largest)
    precomputed_indices = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long, device=q.device
    )

    # Gather components to pack for each quaternion
    pack_indices = precomputed_indices[largest_components]
    components_to_pack = q[batch_indices[:, None], pack_indices]

    # Scale and pack each component into 10-bit integers
    norm = math.sqrt(2) * 0.5
    scaled = components_to_pack * norm + 0.5
    packed = pack_unorm(scaled, 10)  # Assuming pack_unorm is vectorized

    # Combine into the final 32-bit integer
    largest_packed = largest_components.to(torch.int64) << 30
    c0_packed = packed[:, 0] << 20
    c1_packed = packed[:, 1] << 10
    c2_packed = packed[:, 2]

    result = largest_packed | c0_packed | c1_packed | c2_packed
    return result


def splat2ply_bytes_compressed(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    chunk_max_size: int = 256,
    opacity_threshold: float = 1 / 255,
) -> bytes:
    """Return the binary compressed Ply file. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K*3)
        chunk_max_size (int): Maximum number of splats per chunk. Default: 256
        opacity_threshold (float): Opacity threshold. Default: 1 / 255

    Returns:
        bytes: Binary compressed Ply file representing the model.
    """

    # Filter the splats with too low opacity
    mask = torch.sigmoid(opacities) > opacity_threshold
    means = means[mask]
    scales = scales[mask]
    sh0_colors = sh2rgb(sh0)
    sh0_colors = sh0_colors[mask]
    shN = shN[mask]
    quats = quats[mask]
    opacities = opacities[mask]

    num_splats = means.shape[0]
    n_chunks = num_splats // chunk_max_size + (num_splats % chunk_max_size != 0)
    indices = torch.arange(num_splats)
    indices = sort_centers(means, indices)

    float_properties = [
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
        "min_scale_x",
        "min_scale_y",
        "min_scale_z",
        "max_scale_x",
        "max_scale_y",
        "max_scale_z",
        "min_r",
        "min_g",
        "min_b",
        "max_r",
        "max_g",
        "max_b",
    ]
    uint_properties = [
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    ]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element chunk {n_chunks}\n".encode())
    for prop in float_properties:
        buffer.write(f"property float {prop}\n".encode())
    buffer.write(f"element vertex {num_splats}\n".encode())
    for prop in uint_properties:
        buffer.write(f"property uint {prop}\n".encode())
    buffer.write(f"element sh {num_splats}\n".encode())
    for j in range(shN.shape[1]):
        buffer.write(f"property uchar f_rest_{j}\n".encode())
    buffer.write(b"end_header\n")

    chunk_data = []
    splat_data = []
    sh_data = []
    for chunk_idx in range(n_chunks):
        chunk_end_idx = min((chunk_idx + 1) * chunk_max_size, num_splats)
        chunk_start_idx = chunk_idx * chunk_max_size
        splat_idxs = indices[chunk_start_idx:chunk_end_idx]

        # Bounds
        # Means
        chunk_means = means[splat_idxs]
        min_means = torch.min(chunk_means, dim=0).values
        max_means = torch.max(chunk_means, dim=0).values
        mean_bounds = torch.cat([min_means, max_means])
        # Scales
        chunk_scales = scales[splat_idxs]
        min_scales = torch.min(chunk_scales, dim=0).values
        max_scales = torch.max(chunk_scales, dim=0).values
        min_scales = torch.clamp(min_scales, -20, 20)
        max_scales = torch.clamp(max_scales, -20, 20)
        scale_bounds = torch.cat([min_scales, max_scales])
        # Colors
        chunk_colors = sh0_colors[splat_idxs]
        min_colors = torch.min(chunk_colors, dim=0).values
        max_colors = torch.max(chunk_colors, dim=0).values
        color_bounds = torch.cat([min_colors, max_colors])
        chunk_data.extend([mean_bounds, scale_bounds, color_bounds])

        # Quantized properties:
        # Means
        normalized_means = (chunk_means - min_means) / (max_means - min_means)
        means_i = pack_111011(
            normalized_means[:, 0],
            normalized_means[:, 1],
            normalized_means[:, 2],
        )
        # Quaternions
        chunk_quats = quats[splat_idxs]
        quat_i = pack_rotation(chunk_quats)
        # Scales
        normalized_scales = (chunk_scales - min_scales) / (max_scales - min_scales)
        scales_i = pack_111011(
            normalized_scales[:, 0],
            normalized_scales[:, 1],
            normalized_scales[:, 2],
        )
        # Colors
        normalized_colors = (chunk_colors - min_colors) / (max_colors - min_colors)
        chunk_opacities = opacities[splat_idxs]
        chunk_opacities = 1 / (1 + torch.exp(-chunk_opacities))
        chunk_opacities = chunk_opacities.unsqueeze(-1)
        normalized_colors_i = torch.cat([normalized_colors, chunk_opacities], dim=-1)
        color_i = pack_8888(
            normalized_colors_i[:, 0],
            normalized_colors_i[:, 1],
            normalized_colors_i[:, 2],
            normalized_colors_i[:, 3],
        )
        splat_data_chunk = torch.stack([means_i, quat_i, scales_i, color_i], dim=1)
        splat_data_chunk = splat_data_chunk.ravel().to(torch.int64)
        splat_data.extend([splat_data_chunk])

        # Quantized spherical harmonics
        shN_chunk = shN[splat_idxs]
        shN_chunk_quantized = (shN_chunk / 8 + 0.5) * 256
        shN_chunk_quantized = torch.clamp(torch.trunc(shN_chunk_quantized), 0, 255)
        shN_chunk_quantized = shN_chunk_quantized.to(torch.uint8)
        sh_data.extend([shN_chunk_quantized.ravel()])

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    uint32_dtype = np.dtype(np.uint32).newbyteorder("<")
    uint8_dtype = np.dtype(np.uint8)

    buffer.write(
        torch.cat(chunk_data).detach().cpu().numpy().astype(float_dtype).tobytes()
    )
    buffer.write(
        torch.cat(splat_data).detach().cpu().numpy().astype(uint32_dtype).tobytes()
    )
    buffer.write(
        torch.cat(sh_data).detach().cpu().numpy().astype(uint8_dtype).tobytes()
    )

    return buffer.getvalue()


def splat2ply_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,
) -> bytes:
    """Return the binary Ply file. Supported by almost all viewers.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K*3)

    Returns:
        bytes: Binary Ply file representing the model.
    """

    num_splats = means.shape[0]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {num_splats}\n".encode())
    buffer.write(b"property float x\n")
    buffer.write(b"property float y\n")
    buffer.write(b"property float z\n")
    for i, data in enumerate([sh0, shN]):
        prefix = "f_dc" if i == 0 else "f_rest"
        if data is not None:
            for j in range(data.shape[1]):
                buffer.write(f"property float {prefix}_{j}\n".encode())
    buffer.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    # Concatenate all tensors in the correct order
    if shN is not None:
        splat_data = torch.cat(
            [means, sh0, shN, opacities.unsqueeze(1), scales, quats], dim=1
        )
    else:
        splat_data = torch.cat(
            [means, sh0, opacities.unsqueeze(1), scales, quats], dim=1
        )
    # Ensure correct dtype
    splat_data = splat_data.to(torch.float32)

    # Write binary data
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    buffer.write(splat_data.detach().cpu().numpy().astype(float_dtype).tobytes())

    return buffer.getvalue()


def splat2splat_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
) -> bytes:
    """Return the binary Splat file. Supported by antimatter15 viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)

    Returns:
        bytes: Binary Splat file representing the model.
    """

    # Preprocess
    scales = torch.exp(scales)
    sh0_color = sh2rgb(sh0)
    colors = torch.cat([sh0_color, torch.sigmoid(opacities).unsqueeze(-1)], dim=1)
    colors = (colors * 255).clamp(0, 255).to(torch.uint8)

    rots = (quats / torch.linalg.norm(quats, dim=1, keepdim=True)) * 128 + 128
    rots = rots.clamp(0, 255).to(torch.uint8)

    # Sort splats
    num_splats = means.shape[0]
    indices = sort_centers(means, torch.arange(num_splats))

    # Reorder everything
    means = means[indices]
    scales = scales[indices]
    colors = colors[indices]
    rots = rots[indices]

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    means_np = means.detach().cpu().numpy().astype(float_dtype)
    scales_np = scales.detach().cpu().numpy().astype(float_dtype)
    colors_np = colors.detach().cpu().numpy().astype(np.uint8)
    rots_np = rots.detach().cpu().numpy().astype(np.uint8)

    buffer = BytesIO()
    for i in range(num_splats):
        buffer.write(means_np[i].tobytes())
        buffer.write(scales_np[i].tobytes())
        buffer.write(colors_np[i].tobytes())
        buffer.write(rots_np[i].tobytes())

    return buffer.getvalue()


def export_splats(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,
    format: Literal["ply", "splat", "ply_compressed"] = "ply",
    save_to: Optional[str] = None,
) -> bytes:
    """Export a Gaussian Splats model to bytes.
    The three supported formats are:
    - ply: A standard PLY file format. Supported by most viewers.
    - splat: A custom Splat file format. Supported by antimatter15 viewer.
    - ply_compressed: A compressed PLY file format. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
        format (str): Export format. Options: "ply", "splat", "ply_compressed". Default: "ply"
        save_to (str): Output file path. If provided, the bytes will be written to file.
    """
    total_splats = means.shape[0]
    assert means.shape == (total_splats, 3), "Means must be of shape (N, 3)"
    assert scales.shape == (total_splats, 3), "Scales must be of shape (N, 3)"
    assert quats.shape == (total_splats, 4), "Quaternions must be of shape (N, 4)"
    assert opacities.shape == (total_splats,), "Opacities must be of shape (N,)"
    assert sh0.shape == (total_splats, 1, 3), "sh0 must be of shape (N, 1, 3)"
    if shN is not None:
        assert (
        shN.ndim == 3 and shN.shape[0] == total_splats and shN.shape[2] == 3
    ), f"shN must be of shape (N, K, 3), got {shN.shape}"

    # Reshape spherical harmonics
    sh0 = sh0.squeeze(1)  # Shape (N, 3)
    if shN is not None:
        shN = shN.permute(0, 2, 1).reshape(means.shape[0], -1)  # Shape (N, K * 3)

    # Check for NaN or Inf values
    if shN is not None:
        invalid_mask = (
            torch.isnan(means).any(dim=1)
            | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities).any(dim=0)
        | torch.isinf(opacities).any(dim=0)
        | torch.isnan(sh0).any(dim=1)
        | torch.isinf(sh0).any(dim=1)
        | torch.isnan(shN).any(dim=1)
        | torch.isinf(shN).any(dim=1)
        )
    else:
        invalid_mask = (
            torch.isnan(means).any(dim=1)
            | torch.isinf(means).any(dim=1)
            | torch.isnan(scales).any(dim=1)
            | torch.isinf(scales).any(dim=1)
            | torch.isnan(quats).any(dim=1)
            | torch.isinf(quats).any(dim=1)
            | torch.isnan(opacities).any(dim=0)
            | torch.isinf(opacities).any(dim=0)
            | torch.isnan(sh0).any(dim=1)
            | torch.isinf(sh0).any(dim=1)
        )

    # Filter out invalid entries
    valid_mask = ~invalid_mask
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    if shN is not None:
        shN = shN[valid_mask]

    if format == "ply":
        data = splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
    elif format == "splat":
        data = splat2splat_bytes(means, scales, quats, opacities, sh0)
    elif format == "ply_compressed":
        data = splat2ply_bytes_compressed(means, scales, quats, opacities, sh0, shN)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if save_to:
        with open(save_to, "wb") as binary_file:
            binary_file.write(data)

    return data


def rgb2sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to Sphere Harmonics

    Args:
        rgb (torch.Tensor): RGB tensor

    Returns:
        torch.Tensor: SH tensor
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def unpack_unorm(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack an unsigned integer into a floating point value.

    Args:
        packed (torch.Tensor): Packed integer values. Shape (N,)
        bits (int): Number of bits that were packed.

    Returns:
        torch.Tensor: Unpacked floating point values. Shape (N,)
    """
    t = (1 << bits) - 1
    return packed.float() / t


def unpack_111011(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a 32-bit integer into three floating point values with 11, 10, and 11 bits.

    Args:
        packed (torch.Tensor): Packed values. Shape (N,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Unpacked x, y, z components. Each Shape (N,)
    """
    # Extract components using bitwise operations
    packed_z = packed & 0x7FF  # 11 bits
    packed_y = (packed >> 11) & 0x3FF  # 10 bits
    packed_x = (packed >> 21) & 0x7FF  # 11 bits
    
    # Unpack each component
    x = unpack_unorm(packed_x, 11)
    y = unpack_unorm(packed_y, 10)
    z = unpack_unorm(packed_z, 11)
    
    return x, y, z


def unpack_8888(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a 32-bit integer into four floating point values with 8 bits each.

    Args:
        packed (torch.Tensor): Packed values. Shape (N,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Unpacked x, y, z, w components. Each Shape (N,)
    """
    # Extract components using bitwise operations
    packed_w = packed & 0xFF  # 8 bits
    packed_z = (packed >> 8) & 0xFF  # 8 bits
    packed_y = (packed >> 16) & 0xFF  # 8 bits
    packed_x = (packed >> 24) & 0xFF  # 8 bits
    
    # Unpack each component
    x = unpack_unorm(packed_x, 8)
    y = unpack_unorm(packed_y, 8)
    z = unpack_unorm(packed_z, 8)
    w = unpack_unorm(packed_w, 8)
    
    return x, y, z, w


def unpack_rotation(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a 32-bit integer into a quaternion.

    Args:
        packed (torch.Tensor): Packed values. Shape (N,)

    Returns:
        torch.Tensor: Quaternions. Shape (N, 4)
    """
    # Extract the largest component index and packed components
    largest_components = (packed >> 30).to(torch.long)  # 2 bits
    c0_packed = (packed >> 20) & 0x3FF  # 10 bits
    c1_packed = (packed >> 10) & 0x3FF  # 10 bits
    c2_packed = packed & 0x3FF  # 10 bits
    
    # Unpack components
    norm = math.sqrt(2) * 0.5
    c0 = unpack_unorm(c0_packed, 10) - 0.5
    c1 = unpack_unorm(c1_packed, 10) - 0.5
    c2 = unpack_unorm(c2_packed, 10) - 0.5
    c0 = c0 / norm
    c1 = c1 / norm
    c2 = c2 / norm
    
    # Reconstruct quaternions
    batch_size = packed.size(0)
    q = torch.zeros((batch_size, 4), device=packed.device, dtype=torch.float32)
    
    # Precomputed indices for placing components back
    precomputed_indices = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long, device=packed.device
    )
    
    batch_indices = torch.arange(batch_size, device=packed.device)
    pack_indices = precomputed_indices[largest_components]
    
    # Place the three components back
    q[batch_indices[:, None], pack_indices] = torch.stack([c0, c1, c2], dim=1)
    
    # Compute the largest component using the constraint that |q| = 1
    sum_of_squares = (q ** 2).sum(dim=1, keepdim=True)
    largest_values = torch.sqrt(torch.clamp(1.0 - sum_of_squares, min=0.0))
    q[batch_indices, largest_components] = largest_values.squeeze()
    
    return q


def load_ply_bytes(data: bytes) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load splats from binary PLY data.

    Args:
        data (bytes): Binary PLY file data.

    Returns:
        Tuple: (means, scales, quats, opacities, sh0, shN)
            - means (torch.Tensor): Splat means. Shape (N, 3)
            - scales (torch.Tensor): Splat scales. Shape (N, 3)
            - quats (torch.Tensor): Splat quaternions. Shape (N, 4)
            - opacities (torch.Tensor): Splat opacities. Shape (N,)
            - sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
            - shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
    """
    buffer = BytesIO(data)
    
    # Parse header
    line = buffer.readline().decode().strip()
    if line != "ply":
        raise ValueError("Not a valid PLY file")
    
    line = buffer.readline().decode().strip()
    if line != "format binary_little_endian 1.0":
        raise ValueError("Only binary little endian PLY files are supported")
    
    # Parse elements and properties
    elements = {}
    current_element = None
    
    while True:
        line = buffer.readline().decode().strip()
        if line == "end_header":
            break
        
        parts = line.split()
        if parts[0] == "element":
            element_name = parts[1]
            element_count = int(parts[2])
            elements[element_name] = {"count": element_count, "properties": []}
            current_element = element_name
        elif parts[0] == "property":
            if current_element:
                prop_type = parts[1]
                prop_name = parts[2]
                elements[current_element]["properties"].append((prop_type, prop_name))
    
    # Check if this is a compressed PLY
    is_compressed = "chunk" in elements and "vertex" in elements and "sh" in elements
    
    if is_compressed:
        return _load_compressed_ply_bytes(buffer, elements)
    else:
        return _load_standard_ply_bytes(buffer, elements)


def _load_standard_ply_bytes(buffer: BytesIO, elements: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load splats from standard PLY format."""
    if "vertex" not in elements:
        raise ValueError("PLY file must contain vertex element")
    
    vertex_count = elements["vertex"]["count"]
    properties = elements["vertex"]["properties"]
    
    # Calculate data size per vertex
    bytes_per_vertex = sum(4 for prop_type, _ in properties if prop_type == "float")
    total_bytes = vertex_count * bytes_per_vertex
    
    # Read all vertex data
    vertex_data = buffer.read(total_bytes)
    if len(vertex_data) != total_bytes:
        raise ValueError("Incomplete PLY file")
    
    # Parse vertex data
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    vertex_array = np.frombuffer(vertex_data, dtype=float_dtype).reshape(vertex_count, -1).copy()
    vertex_tensor = torch.from_numpy(vertex_array).float()
    
    # Extract components based on property order
    prop_idx = 0
    
    # Means (x, y, z)
    means = vertex_tensor[:, prop_idx:prop_idx+3]
    prop_idx += 3
    
    # Count SH properties
    sh0_count = sum(1 for _, name in properties if name.startswith("f_dc_"))
    shN_count = sum(1 for _, name in properties if name.startswith("f_rest_"))
    
    # SH coefficients
    sh0_data = vertex_tensor[:, prop_idx:prop_idx+sh0_count]
    prop_idx += sh0_count
    
    if shN_count > 0:
        shN_data = vertex_tensor[:, prop_idx:prop_idx+shN_count]
        prop_idx += shN_count
    else:
        shN_data = torch.zeros((vertex_count, 0), device=vertex_tensor.device)
    
    # Opacity
    opacities = vertex_tensor[:, prop_idx]
    prop_idx += 1
    
    # Scales
    scales = vertex_tensor[:, prop_idx:prop_idx+3]
    prop_idx += 3
    
    # Quaternions
    quats = vertex_tensor[:, prop_idx:prop_idx+4]
    
    # Convert SH to proper format
    sh0 = sh0_data.reshape(vertex_count, 1, 3)
    if shN_count > 0:
        K = shN_count // 3
        shN = shN_data.reshape(vertex_count, 3, K).permute(0, 2, 1)
    else:
        shN = torch.zeros((vertex_count, 0, 3), device=vertex_tensor.device)
    
    return means, scales, quats, opacities, sh0, shN


def _load_compressed_ply_bytes(buffer: BytesIO, elements: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load splats from compressed PLY format."""
    chunk_count = elements["chunk"]["count"]
    vertex_count = elements["vertex"]["count"]
    sh_count = elements["sh"]["count"]
    
    # Read chunk data (bounds)
    chunk_properties = elements["chunk"]["properties"]
    float_props_per_chunk = len([p for p in chunk_properties if p[0] == "float"])
    chunk_bytes = chunk_count * float_props_per_chunk * 4
    
    chunk_data = buffer.read(chunk_bytes)
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    chunk_array = np.frombuffer(chunk_data, dtype=float_dtype).reshape(chunk_count, float_props_per_chunk).copy()
    chunk_tensor = torch.from_numpy(chunk_array).float()
    
    # Read vertex data (packed)
    vertex_properties = elements["vertex"]["properties"]
    uint_props_per_vertex = len([p for p in vertex_properties if p[0] == "uint"])
    vertex_bytes = vertex_count * uint_props_per_vertex * 4
    
    vertex_data = buffer.read(vertex_bytes)
    uint32_dtype = np.dtype(np.uint32).newbyteorder("<")
    vertex_array = np.frombuffer(vertex_data, dtype=uint32_dtype).reshape(vertex_count, uint_props_per_vertex).copy()
    vertex_tensor = torch.from_numpy(vertex_array).long()
    
    # Read SH data
    sh_properties = elements["sh"]["properties"]
    sh_props_per_entry = len([p for p in sh_properties if p[0] == "uchar"])
    sh_bytes = sh_count * sh_props_per_entry
    
    sh_data = buffer.read(sh_bytes)
    sh_array = np.frombuffer(sh_data, dtype=np.uint8).reshape(sh_count, sh_props_per_entry).copy()
    sh_tensor = torch.from_numpy(sh_array)
    
    # Determine chunk size
    chunk_max_size = min(256, vertex_count // chunk_count + (1 if vertex_count % chunk_count != 0 else 0))
    
    # Unpack data chunk by chunk
    means_list = []
    scales_list = []
    quats_list = []
    opacities_list = []
    sh0_list = []
    shN_list = []
    
    vertex_idx = 0
    sh_idx = 0
    
    for chunk_idx in range(chunk_count):
        chunk_end_idx = min(vertex_idx + chunk_max_size, vertex_count)
        chunk_size = chunk_end_idx - vertex_idx
        
        if chunk_size <= 0:
            break
        
        # Get chunk bounds
        chunk_bounds = chunk_tensor[chunk_idx]
        
        # Extract bounds
        min_means = chunk_bounds[:3]
        max_means = chunk_bounds[3:6]
        min_scales = chunk_bounds[6:9]
        max_scales = chunk_bounds[9:12]
        min_colors = chunk_bounds[12:15]
        max_colors = chunk_bounds[15:18]
        
        # Get packed vertex data for this chunk
        chunk_vertex_data = vertex_tensor[vertex_idx:chunk_end_idx]
        
        # Unpack means
        packed_means = chunk_vertex_data[:, 0]
        norm_x, norm_y, norm_z = unpack_111011(packed_means)
        chunk_means = torch.stack([norm_x, norm_y, norm_z], dim=1)
        chunk_means = chunk_means * (max_means - min_means) + min_means
        
        # Unpack quaternions
        packed_quats = chunk_vertex_data[:, 1]
        chunk_quats = unpack_rotation(packed_quats)
        
        # Unpack scales
        packed_scales = chunk_vertex_data[:, 2]
        norm_sx, norm_sy, norm_sz = unpack_111011(packed_scales)
        chunk_scales = torch.stack([norm_sx, norm_sy, norm_sz], dim=1)
        chunk_scales = chunk_scales * (max_scales - min_scales) + min_scales
        
        # Unpack colors and opacity
        packed_colors = chunk_vertex_data[:, 3]
        norm_r, norm_g, norm_b, norm_a = unpack_8888(packed_colors)
        chunk_colors = torch.stack([norm_r, norm_g, norm_b], dim=1)
        chunk_colors = chunk_colors * (max_colors - min_colors) + min_colors
        chunk_opacities = norm_a
        
        # Convert colors back to SH
        chunk_sh0 = rgb2sh(chunk_colors)
        
        # Convert opacity back to logit space
        chunk_opacities = torch.clamp(chunk_opacities, 1e-6, 1 - 1e-6)
        chunk_opacities = torch.logit(chunk_opacities)
        
        # Get SH data for this chunk
        chunk_sh_data = sh_tensor[sh_idx:sh_idx + chunk_size]
        chunk_shN = (chunk_sh_data.float() / 256.0 - 0.5) * 8
        
        means_list.append(chunk_means)
        scales_list.append(chunk_scales)
        quats_list.append(chunk_quats)
        opacities_list.append(chunk_opacities)
        sh0_list.append(chunk_sh0)
        shN_list.append(chunk_shN)
        
        vertex_idx = chunk_end_idx
        sh_idx += chunk_size
    
    # Concatenate all chunks
    means = torch.cat(means_list, dim=0)
    scales = torch.cat(scales_list, dim=0)
    quats = torch.cat(quats_list, dim=0)
    opacities = torch.cat(opacities_list, dim=0)
    sh0 = torch.cat(sh0_list, dim=0)
    shN = torch.cat(shN_list, dim=0)
    
    # Reshape SH data
    sh0 = sh0.unsqueeze(1)  # Shape (N, 1, 3)
    if shN.shape[1] > 0:
        K = shN.shape[1] // 3
        shN = shN.reshape(means.shape[0], K, 3)
    else:
        shN = torch.zeros((means.shape[0], 0, 3), device=means.device)
    
    return means, scales, quats, opacities, sh0, shN


def import_splats(
    file_path: str,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load splats from a PLY file.

    Args:
        file_path (str): Path to the PLY file.
        device (str): Device to load tensors on. Default: "cuda"

    Returns:
        Tuple: (means, scales, quats, opacities, sh0, shN)
            - means (torch.Tensor): Splat means. Shape (N, 3)
            - scales (torch.Tensor): Splat scales. Shape (N, 3)
            - quats (torch.Tensor): Splat quaternions. Shape (N, 4)
            - opacities (torch.Tensor): Splat opacities. Shape (N,)
            - sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
            - shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
    """
    with open(file_path, "rb") as f:
        data = f.read()
    
    means, scales, quats, opacities, sh0, shN = load_ply_bytes(data)
    
    # Move to specified device
    means = means.to(device)
    scales = scales.to(device)
    quats = quats.to(device)
    opacities = opacities.to(device)
    sh0 = sh0.to(device)
    shN = shN.to(device)
    
    return means, scales, quats, opacities, sh0, shN

def save_ply(splats: torch.nn.ParameterDict, path: str):
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()

    def construct_list_of_attributes(splats):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(filename, device="cuda"):
    """Load 3D Gaussian model data from PLY file.

    Args:
        filename: Input PLY file path
        device: Computation device

    Returns:
        Tuple: (means, scales, quats, opacities, sh0, shN)
            - means (torch.Tensor): Splat means. Shape (N, 3)
            - scales (torch.Tensor): Splat scales. Shape (N, 3)
            - quats (torch.Tensor): Splat quaternions. Shape (N, 4)
            - opacities (torch.Tensor): Splat opacities. Shape (N,)
            - sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
            - shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
    """
    print(f"Loading Gaussian splats from {filename}")

    # Load PLY file
    plydata = PlyData.read(filename)
    vertices = plydata["vertex"].data

    # Extract properties (positions, normals, features, opacity, scales, rotations)
    # We know the first 6 elements are x,y,z and nx,ny,nz
    xyz = np.vstack([vertices[name] for name in ["x", "y", "z"]]).T

    # Find fields by prefix
    f_dc_fields = [name for name in vertices.dtype.names if name.startswith("f_dc_")]
    f_rest_fields = [
        name for name in vertices.dtype.names if name.startswith("f_rest_")
    ]
    scale_fields = [
        name for name in vertices.dtype.names if name.startswith("scale_")
    ]
    rot_fields = [name for name in vertices.dtype.names if name.startswith("rot_")]

    # Extract features
    f_dc = (
        np.vstack([vertices[name] for name in f_dc_fields]).T
        if f_dc_fields
        else np.zeros((len(xyz), 0))
    )
    f_rest = (
        np.vstack([vertices[name] for name in f_rest_fields]).T
        if f_rest_fields
        else None
    )

    # Extract other properties
    opacities = vertices["opacity"]  # Already in logit space
    scales = np.vstack(
        [vertices[name] for name in scale_fields]
    ).T  # Already in log space
    rotations = np.vstack([vertices[name] for name in rot_fields]).T

    # Create dictionary to hold Gaussian parameters
    gaussian_data = {}

    # Convert to torch tensors and add to dictionary
    gaussian_data["means"] = torch.tensor(xyz, dtype=torch.float32, device=device)

    # Determine if this is SH or feature-based
    if len(f_dc_fields) == 3:
        # This is likely SH-based
        # Reshape f_dc and f_rest to original format: [N, 3K] -> [N, K, 3]
        sh0_dim = 1  # First SH band has 1 coefficient
        f_dc_reshaped = f_dc.reshape(-1, 3, sh0_dim).transpose(0, 2, 1)

        # Number of SH bands in f_rest
        if f_rest is not None:
            rest_coeffs = len(f_rest_fields) // 3
            f_rest_reshaped = f_rest.reshape(-1, 3, rest_coeffs).transpose(0, 2, 1)
            gaussian_data["sh0"] = torch.tensor(
                f_dc_reshaped, dtype=torch.float32, device=device
            )
            gaussian_data["shN"] = torch.tensor(
                f_rest_reshaped, dtype=torch.float32, device=device
            )
        else:
            # Only SH0 (DC term)
            gaussian_data["sh0"] = torch.tensor(
                f_dc_reshaped, dtype=torch.float32, device=device
            )
            gaussian_data["shN"] = torch.zeros(
                (len(xyz), 0, 3), dtype=torch.float32, device=device
            )
    elif len(f_dc_fields) > 0:
        # This is feature-based
        gaussian_data["colors"] = torch.tensor(f_dc, dtype=torch.float32, device=device)
        if f_rest is not None:
            gaussian_data["features"] = torch.tensor(
                f_rest, dtype=torch.float32, device=device
            )

    # Add other properties
    # IMPORTANT: opacities are already in logit space (inverse sigmoid)
    gaussian_data["opacities"] = torch.tensor(
        opacities, dtype=torch.float32, device=device
    )

    # IMPORTANT: scales are already in log space - don't apply log again!
    gaussian_data["scales"] = torch.tensor(scales, dtype=torch.float32, device=device)

    # Add rotations (quaternions)
    gaussian_data["quats"] = torch.tensor(
        rotations, dtype=torch.float32, device=device
    )

    # For debugging
    # Print min/max values
    print(f"Scales min/max: {scales.min():.4f}/{scales.max():.4f}")
    print(f"Opacities min/max: {opacities.min():.4f}/{opacities.max():.4f}")

    print(
        f"Loaded {len(xyz)} Gaussian splats with {len(f_dc_fields)} DC features and "
        f"{len(f_rest_fields) if f_rest is not None else 0} rest features"
    )

    means = gaussian_data["means"]
    scales = gaussian_data["scales"]
    quats = gaussian_data["quats"]
    opacities = gaussian_data["opacities"]

    if "sh0" in gaussian_data:
        sh0 = gaussian_data["sh0"]
    elif "colors" in gaussian_data:
        sh0 = gaussian_data["colors"].unsqueeze(1)
    else:
        sh0 = torch.zeros((len(means), 1, 3), dtype=torch.float32, device=device)

    if "shN" in gaussian_data:
        shN = gaussian_data["shN"]
    else:
        shN = torch.zeros((len(means), 0, 3), dtype=torch.float32, device=device)

    return means, scales, quats, opacities, sh0, shN
