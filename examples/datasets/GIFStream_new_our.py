import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from .read_write_model import read_model, qvec2rotmat

import math

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser with configurable dataset format."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        first_frame: int = 1,
        # New configurable parameters
        colmap_subdir: str = "sparse/0",  # COLMAP directory relative to data_dir
        image_subdir: str = "images",        # Image directory relative to data_dir
        image_format: str = "jpg",        # Image format: "png", "jpg", etc.
        frame_format: str = "06d",        # Frame number format: "05d", "06d", etc.
        frame_offset: int = 1,            # Frame number offset
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.first_frame = first_frame
        
        # Store configurable parameters
        self.colmap_subdir = colmap_subdir
        self.image_subdir = image_subdir
        self.image_format = image_format
        self.frame_format = frame_format
        self.frame_offset = frame_offset
        
        # Try different COLMAP directory patterns
        colmap_dir = os.path.join(data_dir, self.colmap_subdir)
        if not os.path.exists(colmap_dir):
            # Try alternative patterns
            alt_patterns = [
                os.path.join(data_dir, "sparse", "0"),
                os.path.join(data_dir, "colmap_04", "sparse", "0"),
                os.path.join(data_dir, "colmap_06", "sparse", "0"),
                os.path.join(data_dir, "colmap", "sparse", "0"),
            ]
            for alt_dir in alt_patterns:
                if os.path.exists(alt_dir):
                    colmap_dir = alt_dir
                    print(f"[Parser] Using alternative COLMAP directory: {colmap_dir}")
                    break
        
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist. Tried: {[colmap_dir] + alt_patterns}"

        cameras, images, points3D = read_model(path=colmap_dir)
        
        # Group images by camera base name (e.g., '001-000/00000.png -> '001-000')
        images_by_base = {}
        for img in images.values():
            # Handle format like "001-000/00000.png" - extract camera folder before slash
            if '/' in img.name:
                base_name = img.name.split('/')[0]  # Get camera folder (e.g., "001-000")
            elif '_' in img.name:
                # Fallback to old format: remove extension and split by underscore
                base_name = img.name.replace('.png', '').replace('.jpg', '')#.split('_')[0]
            else:
                base_name = img.name.replace('.jpg', '')
            if base_name not in images_by_base:
                images_by_base[base_name] = []
            images_by_base[base_name].append(img)
            
        # Get sorted list of unique camera bases, which defines our cameras
        unique_camera_bases = sorted(images_by_base.keys())
        
        w2c_mats = []
        camera_ids = []
        image_names = []
        
        # For each camera base, select a representative image to get the pose
        for base in unique_camera_bases:
            # Sort images by name and pick the first as representative
            representative_image = sorted(images_by_base[base], key=lambda i: i.name)[0]
            image_names.append(representative_image.name)
            camera_ids.append(representative_image.camera_id)
            
            # Build w2c matrix from the representative image's pose
            rot = qvec2rotmat(representative_image.qvec)
            trans = representative_image.tvec.reshape(3, 1)
            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
            
        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Process intrinsics for all camera models found in the COLMAP file
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        undist_mask_dict = dict()
        
        for cam_id, cam in cameras.items():
            model_name = cam.model
            if model_name == "SIMPLE_PINHOLE":
                fx, cx, cy = cam.params
                fy = fx
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif model_name == "PINHOLE":
                fx, fy, cx, cy = cam.params[:4]
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif model_name == "SIMPLE_RADIAL":
                f, cx, cy, k1 = cam.params
                fx = fy = f
                params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "RADIAL":
                f, cx, cy, k1, k2 = cam.params
                fx = fy = f
                params = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "OPENCV":
                fx, fy, cx, cy = cam.params[:4]
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "perspective"
            elif model_name == "OPENCV_FISHEYE":
                fx, fy, cx, cy = cam.params[:4]
                params = np.array(cam.params[4:8], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise NotImplementedError(f"Camera model {model_name} not implemented.")

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[cam_id] = K
            params_dict[cam_id] = params
            imsize_dict[cam_id] = (cam.width // factor, cam.height // factor)
            undist_mask_dict[cam_id] = None

        print(
            f"[Parser] Found {len(unique_camera_bases)} unique cameras from {len(images)} images in COLMAP."
        )

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Process 3D points
        sorted_points3D_keys = sorted(points3D.keys())
        point3D_id_to_point3D_idx = {pid: i for i, pid in enumerate(sorted_points3D_keys)}
        points_values = [points3D[k] for k in sorted_points3D_keys]

        points_array = np.array([p.xyz for p in points_values]).astype(np.float32)
        points_errors = np.array([p.error for p in points_values]).astype(np.float32)
        points_colors = np.array([p.rgb for p in points_values]).astype(np.uint8)

        point_indices = dict()
        image_id_to_name_map = {img.name: img_id for img_id, img in images.items()}
        point3D_id_to_images = {p.id: np.column_stack((p.image_ids, p.point2D_idxs)) for p in points3D.values()}

        for p_id in sorted_points3D_keys:
            p = points3D[p_id]
            point_idx = point3D_id_to_point3D_idx[p_id]
            for im_id in p.image_ids:
                if im_id in image_id_to_name_map:
                    image_name = image_id_to_name_map[im_id]
                    point_indices.setdefault(image_name, []).append(point_idx)

        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points_array = transform_points(T1, points_array)

            T2 = align_principle_axes(points_array)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points_array = transform_points(T2, points_array)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        # Finalize attributes for the parser object
        self.image_names = image_names
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.undist_mask_dict = undist_mask_dict
        self.points = points_array
        self.points_err = points_errors
        self.points_rgb = points_colors
        self.point3D_id_to_point3D_idx = point3D_id_to_point3D_idx
        self.point3D_id_to_images = point3D_id_to_images
        self.name_to_image_id = {img.name: img_id for img_id, img in images.items()}
        self.image_id_to_name = image_id_to_name_map
        self.transform = transform
        self.point_indices = point_indices
        
        # Build camera paths using configurable parameters
        # For format like "001-000/00000.png", the camera folder is the base name
        self.campaths = [os.path.join(data_dir, self.image_subdir, base) for base in unique_camera_bases]
        
        # Try to find the first image with configurable format
        first_frame_with_offset = first_frame + self.frame_offset
        first_image_path = os.path.join(self.campaths[0], f"{first_frame_with_offset:{self.frame_format}}.{self.image_format}")
        
        if not os.path.exists(first_image_path):
            # Try alternative patterns
            alt_patterns = [
                os.path.join(self.campaths[0], f"{first_frame:06d}.png"),
                os.path.join(self.campaths[0], f"{first_frame:05d}.png"),
                os.path.join(self.campaths[0], f"{first_frame:06d}.jpg"),
                os.path.join(self.campaths[0], f"{first_frame:05d}.jpg"),
            ]
            for alt_path in alt_patterns:
                if os.path.exists(alt_path):
                    first_image_path = alt_path
                    print(f"[Parser] Using alternative image path: {first_image_path}")
                    break
        
        if not os.path.exists(first_image_path):
            raise FileNotFoundError(f"Could not find image to check dimensions: {first_image_path}")
        
        actual_image = imageio.imread(first_image_path)[..., :3]

        actual_height, actual_width = [x // factor for x in actual_image.shape[:2]]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        if actual_height != colmap_height or actual_width != colmap_width:
            print("[Parser] Mismatch between COLMAP and actual image size. Applying scaling.")
            s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
            for cam_id, K in self.Ks_dict.items():
                K[0, :] *= s_width
                K[1, :] *= s_height
                self.Ks_dict[cam_id] = K
                width, height = self.imsize_dict[cam_id]
                self.imsize_dict[cam_id] = (int(width * s_width), int(height * s_height))

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # --- DEBUGGING SECTION ---
        print("\n--- Parser Debug Info ---")
        print(f"Dataset Configuration:")
        print(f"  - COLMAP subdir: {self.colmap_subdir}")
        print(f"  - Image subdir: {self.image_subdir}")
        print(f"  - Image format: {self.image_format}")
        print(f"  - Frame format: {self.frame_format}")
        print(f"  - Frame offset: {self.frame_offset}")
        print(f"Found {len(self.campaths)} unique cameras.")
        if self.campaths:
            print(f"Example camera path: {self.campaths[0]}")
        print(f"Loaded {len(self.camtoworlds)} camera poses.")
        
        for i in range(len(self.campaths)):
            print(f"\n--- DEBUG: Camera {i} ---")
            
            # Get info for this unique camera
            cam_id_for_intrinsics = self.camera_ids[i]
            rep_image_name = self.image_names[i]
            
            # Find the original image object to get qvec
            image_id = self.name_to_image_id[rep_image_name]
            image_obj = images[image_id]
            qvec = image_obj.qvec

            # Intrinsics
            K = self.Ks_dict[cam_id_for_intrinsics]
            
            # Associated points
            point_idxs = self.point_indices.get(rep_image_name)
            
            print(f"  Camera ID (for intrinsics): {cam_id_for_intrinsics}")
            print(f"  Representative Image Name: {rep_image_name}")
            print(f"  Pose (qvec): [qw={qvec[0]:.4f}, qx={qvec[1]:.4f}, qy={qvec[2]:.4f}, qz={qvec[3]:.4f}]")
            print(f"  Intrinsics (K):\n{K}")
            
            if point_idxs is not None and len(point_idxs) > 0:
                print(f"  Associated with {len(point_idxs)} 3D points.")
                # Print first 5 associated points
                num_points_to_show = min(5, len(point_idxs))
                for j in range(num_points_to_show):
                    point_idx = point_idxs[j]
                    point_xyz = self.points[point_idx]
                    print(f"    Point {j} (idx={point_idx}): {point_xyz}")
            else:
                print("  No 3D points associated with this representative image.")

        print("--------------------------\n")


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        start_frame: int = 0,
        GOP_size: int = 50,
        test_set: list = [0],
        remove_set: list = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.start_frame = start_frame
        self.GOP_size = GOP_size
        self.cameras_length = len(self.parser.campaths)
        indices = np.arange(self.cameras_length * GOP_size)
        if test_set is None:
            if self.parser.test_every > 0:
                test_set = list(range(0, self.cameras_length, self.parser.test_every))
            else:
                test_set = []
        if split == "train":
            # Use ALL cameras for training (ignore test_set for training)
            # self.indices = [x for x in indices if (x // GOP_size) not in test_set]
            self.indices = list(indices)
        else:
            # Use only test_set cameras for validation/testing
            self.indices = [x for x in indices if (x // GOP_size) in test_set]

        if remove_set is not None:
            self.indices = [x for x in self.indices if (x // GOP_size) not in remove_set]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cam_idx = self.indices[item] // self.GOP_size
        frame_idx = self.start_frame + (self.indices[item] % self.GOP_size)
        
        camera_path = self.parser.campaths[cam_idx]
        
        # Use configurable frame format and image format
        frame_with_offset = frame_idx + self.parser.frame_offset
        image_path = os.path.join(camera_path, f"{frame_with_offset:{self.parser.frame_format}}.{self.parser.image_format}")
        
        # Try alternative formats if the configured one doesn't exist
        if not os.path.exists(image_path):
            alt_formats = [
                f"{frame_idx:06d}.png",
                f"{frame_idx:05d}.png",
                f"{frame_idx:06d}.jpg", 
                f"{frame_idx:05d}.jpg",
            ]
            for alt_format in alt_formats:
                alt_path = os.path.join(camera_path, alt_format)
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = imageio.imread(image_path)[..., :3]
        image = cv2.resize(image, dsize=(image.shape[1]//self.parser.factor, image.shape[0]//self.parser.factor), interpolation=cv2.INTER_LINEAR)
        
        # Directly use cam_idx to get the correct pose and intrinsics ID
        camtoworlds = self.parser.camtoworlds[cam_idx]
        camera_id = self.parser.camera_ids[cam_idx]
        
        K = self.parser.Ks_dict[camera_id].copy()

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "time": float(frame_idx - self.start_frame) / (self.GOP_size-1) if self.GOP_size > 1 else 0.0,
            "camera_id": cam_idx,
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="test", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        # The following keys are not available unless load_depths=True is fully implemented
        # for the new parser.
        # points = data["points"].numpy()
        # depths = data["depths"].numpy()
        # for x, y in points:
        #     cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
