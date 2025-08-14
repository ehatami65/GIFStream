import os
import argparse
import time
import yaml
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import glob
from collections import defaultdict
import math
import re
import subprocess

# Use the simple trainer's Runner and Config
from simple_trainer_GIFStream import Runner, Config, quaternion_to_rotation_matrix
from gsplat.compression.compression import Compression
from gsplat.exporter import rgb2sh, export_splats
from utils import find_k_neighbors


@torch.no_grad()
def get_gaussians_for_frame(runner: Runner, time_val: float, previous_gaussians: dict = None):
    """
    Computes and returns the neural Gaussians for a specific time frame.
    Also returns a primitive-level activity mask.
    If previous_gaussians is provided, values for inactive primitives are
    carried forward from previous_gaussians. If not provided, inactive
    primitives are filled with the median value for that parameter.
    """
    cfg = runner.cfg
    device = runner.device
    
    # We always process all anchors for a given GOP's checkpoint.
    visible_anchor_mask = torch.ones(runner.splats["anchors"].shape[0], dtype=torch.bool, device=device)
    
    camtoworlds = torch.eye(4, device=device).unsqueeze(0)
    camera_ids = torch.tensor([0], device=device) if cfg.app_opt else None

    # This logic is adapted from `get_neural_gaussians` and `decoding_features`
    if not cfg.compression_sim or not hasattr(runner, 'comp_sim_splats'):
        selected_anchors = runner.splats["anchors"][visible_anchor_mask]
        selected_offsets = runner.splats["offsets"][visible_anchor_mask]
    else:
        # This part handles models trained with compression simulation, which might be a future use case.
        selected_anchors = runner.comp_sim_splats["anchors"][visible_anchor_mask]
        selected_offsets = runner.comp_sim_splats["offsets"][visible_anchor_mask]

    # --- Start of adapted decoding_features ---
    results = runner.decoding_features(
        camtoworlds, time_val, visible_anchor_mask, canonical=False, step=-1, camera_ids=camera_ids
    )
    # --- End of adapted decoding_features ---

    # --- Start of adapted get_neural_gaussians ---
    neural_opacity = results["neural_opacity"]
    neural_colors = results["neural_colors"]
    neural_scale_rot = results["neural_scale_rot"]
    motion = results["motion"]
    selected_scales = results["selected_scales"]

    # This is the key mask: True for any primitive that should be rendered.
    primitive_activity_mask = (neural_opacity > 0.0).view(-1)
    
    anchor_offset = motion[:, -7:-4]
    moved_anchors = selected_anchors + anchor_offset

    anchor_rot = F.normalize(0.1 * motion[:, -4:] + torch.tensor([[1, 0, 0, 0]], device=device))
    anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)

    transformed_offsets = torch.bmm(
        selected_offsets.view(-1, cfg.n_offsets, 3) * selected_scales.unsqueeze(1)[:, :, :3],
        anchor_rotation.reshape((-1, 3, 3)).transpose(1, 2),
    ).reshape((-1, 3))

    scales_repeated = selected_scales.unsqueeze(1).repeat(1, cfg.n_offsets, 1).view(-1, 6)
    anchors_repeated = moved_anchors.unsqueeze(1).repeat(1, cfg.n_offsets, 1).view(-1, 3)

    scales = scales_repeated[:, 3:] * torch.sigmoid(neural_scale_rot[:, :3])
    scales = torch.log(scales.clamp(min=1e-8))
    quats = F.normalize(neural_scale_rot[:, 3:7])
    means = anchors_repeated + transformed_offsets
    opacities = torch.logit(neural_opacity.squeeze(-1).clamp(min=0.0, max=1.0)).clamp(min=-6, max=12.0)
    
    sh0 = neural_colors.unsqueeze(1)
    # For now, we only handle degree 0 SH.
    shN = torch.zeros((sh0.shape[0], 0, 3), device=device)

    current = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }

    inactive_mask = ~primitive_activity_mask
    if previous_gaussians is not None:
        # Carry forward previous values for inactive primitives
        for key, tensor in current.items():
            prev_tensor = previous_gaussians[key]
            tensor[inactive_mask] = prev_tensor[inactive_mask]
    else:
        # Fill inactive primitives with median per-parameter value
        if inactive_mask.any():
            active_mask = primitive_activity_mask
            # If no active primitives, compute median over all
            def median_over_first_dim(t: torch.Tensor, use_active: bool):
                src = t[active_mask] if use_active and active_mask.any() else t
                if src.ndim == 1:
                    return src.median()
                else:
                    return src.median(dim=0).values

            # means, scales: shape (N, D)
            for key in ["means", "scales"]:
                med = median_over_first_dim(current[key], use_active=True)
                current[key][inactive_mask] = med

            # quats: median then renormalize
            quat_med = median_over_first_dim(current["quats"], use_active=True)
            quat_med = F.normalize(quat_med.unsqueeze(0), dim=-1).squeeze(0)
            current["quats"][inactive_mask] = quat_med

            # sh0: shape (N,1,3); take median over first dim → (1,3)
            sh0_med = median_over_first_dim(current["sh0"], use_active=True)
            current["sh0"][inactive_mask] = sh0_med

            # opacities: shape (N,)
            opa_med = median_over_first_dim(current["opacities"], use_active=True)
            current["opacities"][inactive_mask] = opa_med

    return current, primitive_activity_mask


def write_output(output_dir: str, video_buffers: dict, full_meta: dict):
    """Writes compressed video buffers and metadata to files."""
    codec = "libx265"
    mp4_pixel_format = "rgb24"
    extension = "mp4"

    print(f"--- Writing output files (format: {extension}) ---")
    total_bytes = 0

    for name, frames in video_buffers.items():
        if not frames: continue
        
        output_path = os.path.join(output_dir, f"{name}_dynamic.{extension}")
        print(f"  > Encoding {name} with codec: {codec}, pixel format: {mp4_pixel_format}, container: {extension}")

        h, w = frames[0].shape[:2]
        pad_h, pad_w = (2 - h % 2) % 2, (2 - w % 2) % 2
        
        is_rgb = frames[0].ndim == 3
        input_pix_fmt = 'rgb24' if is_rgb else 'gray8'
        # For libx265, gbrp (planar RGB) is required for lossless RGB output.
        # 'gray' is already planar and works directly.
        output_pix_fmt = 'gbrp' if is_rgb else 'gray'
        
        command = [
            './ffmpeg/bin/ffmpeg',
            '-y',  # Overwrite output file if it exists
            # Input options: describe the raw video stream from the pipe
            '-f', 'rawvideo',
            '-s', f'{w+pad_w}x{h+pad_h}',
            '-pix_fmt', input_pix_fmt,
            '-r', '10',  # Frame rate
            '-i', '-',  # Input from stdin
            # Output options: define the encoding
            '-c:v', codec,
            '-preset', 'medium',
            '-x265-params', 'lossless=1',
            '-pix_fmt', output_pix_fmt,
            output_path
        ]

        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            padded_frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0,0)) if frame.ndim == 3 else ((0, pad_h), (0, pad_w)), 'constant')
            proc.stdin.write(padded_frame.tobytes())
        
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"Error encoding {name}: {stderr.decode('utf-8')}")

        if os.path.exists(output_path):
            total_bytes += os.path.getsize(output_path)
    
    full_meta["video_format"] = extension
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)
    total_bytes += os.path.getsize(meta_path)
    
    print(f"Total compressed size: {total_bytes / 1e6:.2f} MB")


def write_atlas_output(output_dir: str, atlas_frames: list, full_meta: dict, atlas_name: str = "atlas_dynamic"):
    """
    Writes a single atlas video and metadata bundle.
    Atlas frames must be HxWx3 uint8.
    """
    codec = "libx265"
    extension = "mp4"

    output_path = os.path.join(output_dir, f"{atlas_name}.{extension}")

    h, w = atlas_frames[0].shape[:2]
    pad_h, pad_w = (2 - h % 2) % 2, (2 - w % 2) % 2
    
    command = [
        './ffmpeg/bin/ffmpeg',
        '-y',
        # Input options
        '-f', 'rawvideo',
        '-s', f'{w+pad_w}x{h+pad_h}',
        '-pix_fmt', 'rgb24',
        '-r', '10',
        '-i', '-',
        # Output options
        '-c:v', codec,
        '-preset', 'slow',
        '-x265-params', 'lossless=1',
        '-pix_fmt', 'gbrp',
        output_path
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in atlas_frames:
        padded_frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0,0)), 'constant')
        proc.stdin.write(padded_frame.tobytes())
    
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"Error encoding atlas: {stderr.decode('utf-8')}")

    # Write metadata
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)

    size_bytes = 0
    if os.path.exists(output_path):
        size_bytes += os.path.getsize(output_path)
    if os.path.exists(meta_path):
        size_bytes += os.path.getsize(meta_path)
    print(f"Total compressed size: {size_bytes / 1e6:.2f} MB")


def decompress_and_export(output_dir: str, device: str):
    """
    Decompresses data from a unified video stream and exports a full PLY sequence.
    This process is GOP-agnostic.
    """
    print("\n--- Decompressing and Exporting to PLY Sequence ---")
    
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta bundle not found at {meta_path}")

    with open(meta_path, "r") as f:
        full_meta = json.load(f)

    export_dir = os.path.join(output_dir, "decompressed_ply_sequence")
    os.makedirs(export_dir, exist_ok=True)
    
    compressor = Compression(use_sort=False) # Sorting is only for compression

    # 1. Load the unified dynamic video streams
    video_data = defaultdict(list)
    total_frames = full_meta.get("total_frames", 0)
    video_format = "mp4"
    codec = "libx265"
    extension = "mp4"

    # Atlas-aware path
    atlas_meta = full_meta.get("atlas", None)
    if total_frames > 0 and atlas_meta and atlas_meta.get("enabled", False):
        print(f"  Loading atlas video (format: {video_format}, codec: {codec})...")
        atlas_path = os.path.join(output_dir, f"atlas_dynamic.{extension}")
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Atlas video not found at {atlas_path}")
        atlas_frames = imageio.mimread(atlas_path, memtest=False)

        grid_cols = atlas_meta["grid_cols"]
        grid_rows = atlas_meta["grid_rows"]
        tile_size = atlas_meta["tile_size"]
        buffer_specs = atlas_meta["buffer_specs"]  # list of {name, channels}

        # Pre-create containers
        for spec in buffer_specs:
            video_data[spec["name"]] = []

        for frame in atlas_frames:
            for idx, spec in enumerate(buffer_specs):
                row = idx // grid_cols
                col = idx % grid_cols
                y0, y1 = row * tile_size, (row + 1) * tile_size
                x0, x1 = col * tile_size, (col + 1) * tile_size
                tile = frame[y0:y1, x0:x1]
                if spec.get("channels", 3) == 1:
                    # Convert to single channel
                    gray = tile[..., 0]
                    video_data[spec["name"]].append(gray)
                else:
                    video_data[spec["name"]].append(tile)
    else:
        if total_frames > 0:
            print(f"  Loading video streams (format: {video_format}, codec: {codec})...")
            # Infer parameter names from the first frame's metadata
            param_names = set(full_meta["frames_meta"]["0"].keys()) | {"activity_mask"}
            
            for param_name in param_names:
                if param_name == "quats":
                    for component in ['x', 'y', 'z', 'w']:
                        video_path = os.path.join(output_dir, f"quats_{component}_dynamic.{extension}")
                        if os.path.exists(video_path): video_data[f"quats_{component}"] = imageio.mimread(video_path, memtest=False)
                else:
                    for suffix in ["", "_l", "_u"]:
                        video_path = os.path.join(output_dir, f"{param_name}{suffix}_dynamic.{extension}")
                        if os.path.exists(video_path): video_data[f"{param_name}{suffix}"] = imageio.mimread(video_path, memtest=False)
            
            activity_path = os.path.join(output_dir, f"activity_mask_dynamic.{extension}")
            if os.path.exists(activity_path):
                video_data["activity_mask"] = imageio.mimread(activity_path, memtest=False)

    # 2. Decompress and combine frame-by-frame
    for frame_idx in tqdm(range(total_frames), desc="Exporting PLY frames"):
        # This is the metadata template, consistent for all frames
        frame_meta = full_meta["frames_meta"].get(str(frame_idx))
        if not frame_meta: continue
        
        frame_compressed_arrays = {}
        # Get the original, unpadded side length
        max_sidelen = full_meta.get("max_sidelen")

        for param_name in frame_meta.keys():
            if param_name == "quats":
                quat_channels = []
                for component in ['x', 'y', 'z', 'w']:
                    frame = video_data[f"quats_{component}"][frame_idx]
                    if max_sidelen:
                        frame = frame[:max_sidelen, :max_sidelen]
                    quat_channels.append(frame[..., 0] if frame.ndim == 3 else frame)
                frame_compressed_arrays["quats"] = np.stack(quat_channels, axis=-1)
            elif f"{param_name}_l" in video_data:
                frame_l = video_data[f"{param_name}_l"][frame_idx]
                frame_u = video_data[f"{param_name}_u"][frame_idx]
                if max_sidelen:
                    frame_l = frame_l[:max_sidelen, :max_sidelen]
                    frame_u = frame_u[:max_sidelen, :max_sidelen]
                frame_compressed_arrays[f"{param_name}_l"] = frame_l
                frame_compressed_arrays[f"{param_name}_u"] = frame_u
            elif param_name in video_data:
                frame = video_data[param_name][frame_idx]
                if max_sidelen:
                    frame = frame[:max_sidelen, :max_sidelen]
                frame_compressed_arrays[param_name] = frame

        decompressed_padded_splats = compressor.decompress(frame_meta, frame_compressed_arrays, device=device)
        
        # Now, use the activity mask to filter the primitives
        activity_mask_padded = None
        if "activity_mask" in video_data:
            activity_frame_raw = video_data["activity_mask"][frame_idx]
            if max_sidelen:
                activity_frame_raw = activity_frame_raw[:max_sidelen, :max_sidelen]
            activity_frame_gray = activity_frame_raw[..., 0] if activity_frame_raw.ndim == 3 else activity_frame_raw
            activity_mask_padded = (activity_frame_gray.flatten() > 128)
        
        if activity_mask_padded is None:
            print(f"Warning: No activity mask found for frame {frame_idx}. Skipping filtering.")
            final_splats = decompressed_padded_splats
        else:
            final_splats = {}
            for key, tensor in decompressed_padded_splats.items():
                if tensor is not None:
                    # Flatten tensor to (N, D), filter by mask, then restore original shape if needed
                    original_shape = list(tensor.shape)
                    num_primitives = original_shape[0]
                    
                    if num_primitives != len(activity_mask_padded):
                        print(f"Warning: Mismatch in primitive count for param '{key}' in frame {frame_idx}. Skipping filtering.")
                        final_splats[key] = tensor
                        continue

                    filtered_tensor = tensor[activity_mask_padded]
                    if key == "sh0":
                        filtered_tensor = rgb2sh(filtered_tensor)
                    elif key == "opacities":
                        filtered_tensor = torch.sigmoid(filtered_tensor)
                    final_splats[key] = filtered_tensor

        if final_splats.get("means") is not None and final_splats["means"].shape[0] > 0:
            output_ply_path = os.path.join(export_dir, f"frame_{frame_idx:05d}.ply")
            final_splats.pop("shN", None)
            export_splats(save_to=output_ply_path, **final_splats)

    print(f"\nDecompression complete. PLY sequence saved to: {export_dir}")


def load_simple_gop_model(runner: Runner, ckpt_path: str, device: str):
    """Loads a checkpoint from the simple_trainer into the runner."""
    print(f"Loading GOP checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    splats_dict = ckpt.get("splats", {})
    for k in runner.splats.keys():
        if k in splats_dict:
            runner.splats[k].data = splats_dict[k].to(device)
    
    if "decoders" in ckpt:
        runner.decoders.load_state_dict(ckpt["decoders"])
    runner.decoders.to(device)

    # Handle other potential saved states if necessary
    if runner.cfg.app_opt and "app_module" in ckpt:
        runner.app_module.load_state_dict(ckpt["app_module"])

    if runner.cfg.compression_sim:
        runner.cfg.compression_sim = ckpt.get("compression_sim", runner.cfg.compression_sim)
        if runner.cfg.compression_sim:
            runner.load_entropy_model_from_ckpt(ckpt, runner.cfg.entropy_model_type)

    if runner.cfg.knn:
        _, runner.indices = find_k_neighbors(runner.splats["anchors"], runner.cfg.n_knn)


def main():
    parser = argparse.ArgumentParser(description="Compress a trained GIFStream model sequence into a single video stream.")
    parser.add_argument("--mode", required=True, choices=['compress', 'decompress'],
                         help="Operation mode.")
    parser.add_argument("--checkpoints_dir", help="Path to the directory containing GOP checkpoints (for compression modes).")
    parser.add_argument("--config_path", help="Path to the original config.yml file (for compression modes).")
    parser.add_argument("--output_dir", default="./unified_compression_output", help="Directory for compressed/decompressed files.")
    parser.add_argument("--device", default="cuda:0", help="Device to use.")
    parser.add_argument("--use_atlas", action='store_true', help="If set, combine all streams into a single video atlas.")
    parser.add_argument("--atlas_cols", type=int, default=0, help="Number of columns in atlas grid (0=auto sqrt).")
    args = parser.parse_args()

    if args.mode == 'compress':
        if not all([args.checkpoints_dir, args.config_path]):
            parser.error("--checkpoints_dir and --config_path are required for compression mode.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        device = args.device
        
        def gop_sort_key(s):
            # Extracts number from filename for sorting, e.g., GOP_10.pt -> 10
            match = re.search(r'(\d+)', os.path.basename(s))
            return int(match.group(1)) if match else -1

        gop_checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "ckpt_*.pt")), key=gop_sort_key)
        if not gop_checkpoints:
            gop_checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pt")), key=gop_sort_key) # Fallback for different naming
        
        if not gop_checkpoints:
            raise FileNotFoundError(f"No GOP checkpoints found in {args.checkpoints_dir}")

        print("--- Initializing Runner from Config ---")
        with open(args.config_path, 'r') as f:
            # Use safe_load, assuming the config is a simple structure
            trainer_config_dict = yaml.unsafe_load(f)

        cfg = Config()
        for key, value in trainer_config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        cfg.disable_viewer = True
        runner = Runner(0, 0, 1, cfg)

        # --- Pass 1: Generate Locally Optimized Frames for Each GOP ---
        print("--- Pass 1: Caching and processing frames for each GOP locally ---")
        
        processed_gops_data = []
        dynamic_compressor = Compression(use_sort=True, verbose=False)

        for ckpt_path in tqdm(gop_checkpoints, desc="Processing GOPs"):
            load_simple_gop_model(runner, ckpt_path, device)
            
            # a. Cache raw frame data for this GOP
            gop_raw_frames = []
            gop_size = runner.cfg.GOP_size
            previous_gaussians_full = None
            for frame_in_gop in range(gop_size):
                time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
                is_first = (frame_in_gop == 0)
                gaussians, activity_mask = get_gaussians_for_frame(
                    runner, time_val, previous_gaussians=previous_gaussians_full
                )
                gop_raw_frames.append({"gaussians": gaussians, "activity_mask": activity_mask})
                # Update previous for next frame in GOP
                previous_gaussians_full = {k: v.clone() for k, v in gaussians.items()}
            
            # b. Compute GOP-wide activity mask
            if not gop_raw_frames: continue
            gop_wide_activity_mask = torch.stack([f['activity_mask'] for f in gop_raw_frames]).any(dim=0)
            
            num_active_primitives = gop_wide_activity_mask.sum().item()
            if num_active_primitives == 0:
                processed_gops_data.append({'sidelen': 0, 'frames': []})
                continue
            
            # c. Calculate local square size and determine crop count
            n_sidelen_gop = int(math.sqrt(num_active_primitives))
            n_primitives_square = n_sidelen_gop * n_sidelen_gop
            n_crop = num_active_primitives - n_primitives_square
            if n_crop > 0:
                print(f"GOP Info: Cropping {n_crop} primitives to form a {n_sidelen_gop}x{n_sidelen_gop} square.")

            # d. Filter, crop, sort, and store processed frames
            sort_indices = None
            force_resort = True
            gop_processed_frames = []
            keep_indices_mask = None # This will be computed once from the first frame and reused

            for frame_idx, frame_data in enumerate(gop_raw_frames):
                # Filter based on GOP-wide mask first
                filtered_gaussians = {name: tensor[gop_wide_activity_mask] for name, tensor in frame_data["gaussians"].items()}
                filtered_activity_mask = frame_data["activity_mask"][gop_wide_activity_mask]

                # On the first frame, determine which primitives to keep for the whole GOP
                if frame_idx == 0 and n_crop > 0:
                    opacities_for_crop = filtered_gaussians["opacities"]
                    # Get indices of the top N primitives with highest opacity to keep
                    keep_indices_mask = torch.topk(opacities_for_crop, n_primitives_square).indices
                
                # Apply the consistent crop mask to all frames in the GOP
                if keep_indices_mask is not None:
                    cropped_gaussians = {name: tensor[keep_indices_mask] for name, tensor in filtered_gaussians.items()}
                    cropped_activity_mask = filtered_activity_mask[keep_indices_mask]
                else:
                    cropped_gaussians = filtered_gaussians
                    cropped_activity_mask = filtered_activity_mask

                # The data sent to the compressor is now perfectly square, so its internal cropping is skipped.
                frame_meta, frame_arrays, new_indices, _ = dynamic_compressor.compress(
                    cropped_gaussians, sort_indices=sort_indices, force_resort=force_resort
                )
                if new_indices is not None: sort_indices = new_indices
                force_resort = False
                
                # Process the activity mask, which was already cropped, using the same sorting
                sorted_activity_mask = cropped_activity_mask[sort_indices].cpu().numpy()
                activity_grid = sorted_activity_mask.reshape(n_sidelen_gop, n_sidelen_gop)
                frame_arrays["activity_mask"] = (activity_grid * 255).astype(np.uint8)

                gop_processed_frames.append({'arrays': frame_arrays, 'meta': frame_meta})
            
            processed_gops_data.append({'sidelen': n_sidelen_gop, 'frames': gop_processed_frames})

        # --- Pass 2: Unify frame size and Encode Video ---
        print("\n--- Pass 2: Unifying frame sizes and preparing video buffers ---")
        
        if not processed_gops_data:
            print("No data processed. Exiting.")
            return

        first_gop_with_frames = next((gop for gop in processed_gops_data if gop['frames']), None)
        if not first_gop_with_frames:
            print("No frames with data found across all GOPs. Exiting.")
            return
            
        max_sidelen = max(gop['sidelen'] for gop in processed_gops_data)
        print(f"Global video resolution will be {max_sidelen}x{max_sidelen}")

        video_buffers = defaultdict(list)
        final_frames_meta = {}
        
        # Determine all possible buffer names from the first processed frame
        all_possible_buffer_names_set = set(first_gop_with_frames['frames'][0]['arrays'].keys())
        # Manually add separated quat buffers if the combined one exists
        if "quats" in all_possible_buffer_names_set:
            all_possible_buffer_names_set.remove("quats")
            for i in ['x', 'y', 'z', 'w']: all_possible_buffer_names_set.add(f"quats_{i}")

        # Sort buffer names to ensure a consistent order, with means_l first.
        all_possible_buffer_names = sorted(list(all_possible_buffer_names_set))
        if 'means_l' in all_possible_buffer_names:
            all_possible_buffer_names.remove('means_l')
            all_possible_buffer_names.insert(0, 'means_l')


        global_frame_idx = 0
        for gop_data in processed_gops_data:
            sidelen = gop_data['sidelen']
            pad_h = pad_w = max_sidelen - sidelen

            for frame_content in gop_data['frames']:
                frame_arrays = frame_content['arrays']
                
                # Pad each array to the max side length
                for name in all_possible_buffer_names:
                    # Handle quats which are split
                    if name.startswith("quats_"):
                        if "quats" in frame_arrays:
                            quat_idx = ['x', 'y', 'z', 'w'].index(name.split('_')[1])
                            frame = frame_arrays["quats"][..., quat_idx]
                        else: # This GOP had no data, so append a blank frame
                            frame = np.zeros((sidelen, sidelen), dtype=np.uint8)
                    elif name in frame_arrays:
                        frame = frame_arrays[name]
                    else: # This buffer doesn't exist for this GOP (e.g., shN_labels)
                        # Create a blank frame of the correct local size
                        shape = (sidelen, sidelen)
                        if name not in ['activity_mask', 'labels'] and not name.startswith('quats_'):
                            shape += (3,) # Assume RGB if not grayscale
                        frame = np.zeros(shape, dtype=np.uint8)

                    if pad_h > 0 or pad_w > 0:
                        padding = ((0, pad_h), (0, pad_w), (0,0)) if frame.ndim == 3 else ((0, pad_h), (0, pad_w))
                        padded_frame = np.pad(frame, padding, 'constant')
                    else:
                        padded_frame = frame
                    video_buffers[name].append(padded_frame)
                
                final_frames_meta[str(global_frame_idx)] = frame_content['meta']
                global_frame_idx += 1
        
        # We need a single metadata template for the decompressor. 
        # We can take it from the first frame of the first non-empty GOP.
        padded_len = max_sidelen * max_sidelen
        for frame_meta in final_frames_meta.values():
            if frame_meta:
                for param_meta in frame_meta.values():
                    if "shape" in param_meta:
                        param_meta["shape"][0] = padded_len
        
        full_meta = {
            "frames_meta": final_frames_meta,
            "max_sidelen": max_sidelen,
            "total_frames": global_frame_idx,
            "video_format": "mp4",
            "codec": "libx265",
            "mp4_pixel_format": "rgb24",
        }

        if args.use_atlas:
            # Build atlas: determine order and channel specs
            buffer_specs = []
            for name in all_possible_buffer_names:
                if name not in video_buffers: continue
                first_frame = video_buffers[name][0]
                channels = 3 if first_frame.ndim == 3 else 1
                buffer_specs.append({"name": name, "channels": channels})

            n_tiles = len(buffer_specs)
            grid_cols = args.atlas_cols if args.atlas_cols and args.atlas_cols > 0 else int(math.ceil(math.sqrt(n_tiles)))
            grid_rows = int(math.ceil(n_tiles / grid_cols))

            atlas_h = grid_rows * max_sidelen
            atlas_w = grid_cols * max_sidelen

            atlas_frames = []
            for f_idx in range(global_frame_idx):
                atlas = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
                for idx, spec in enumerate(buffer_specs):
                    row = idx // grid_cols
                    col = idx % grid_cols
                    y0, y1 = row * max_sidelen, (row + 1) * max_sidelen
                    x0, x1 = col * max_sidelen, (col + 1) * max_sidelen
                    tile = video_buffers[spec["name"]][f_idx]
                    if tile.ndim == 2:
                        tile_rgb = np.stack([tile, tile, tile], axis=-1)
                    else:
                        tile_rgb = tile
                    atlas[y0:y1, x0:x1] = tile_rgb
                atlas_frames.append(atlas)

            full_meta["atlas"] = {
                "enabled": True,
                "grid_cols": grid_cols,
                "grid_rows": grid_rows,
                "tile_size": max_sidelen,
                "buffer_specs": buffer_specs,
            }
            write_atlas_output(args.output_dir, atlas_frames, full_meta)
        else:
            write_output(args.output_dir, video_buffers, full_meta)

    elif args.mode == 'decompress':
        decompress_and_export(args.output_dir, args.device)

if __name__ == "__main__":
    main()