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

# Use the simple trainer's Runner and Config
from simple_trainer_GIFStream import Runner, Config, quaternion_to_rotation_matrix, quaternion_multiply
from gsplat.compression.compression import Compression
from gsplat.exporter import rgb2sh, export_splats
from utils import find_k_neighbors


@torch.no_grad()
def get_gaussians_for_frame(runner: Runner, time_val: float):
    """
    Computes and returns the neural Gaussians for a specific time frame.
    Also returns a primitive-level activity mask.
    This is adapted from scene_compression.py and the simple_trainer's export logic.
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
    primitive_activity_mask = (neural_opacity >= 0.0).view(-1)
    
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
    quats = F.normalize(neural_scale_rot[:, 3:7])
    means = anchors_repeated + transformed_offsets
    opacities = neural_opacity.squeeze(-1)
    
    colors = neural_colors
    sh0 = rgb2sh(colors).unsqueeze(1)
    # For now, we only handle degree 0 SH.
    shN = torch.zeros((sh0.shape[0], 0, 3), device=device)

    return {
        "means": means, "scales": torch.log(scales.clamp(min=1e-10)), "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN,
    }, primitive_activity_mask


def write_output(output_dir: str, crf: int, video_buffers: dict, full_meta: dict):
    """Writes compressed video buffers and metadata to files."""
    video_format = full_meta.get("video_format", "mp4")
    print(f"--- Writing output files (format: {video_format}) ---")
    total_bytes = 0

    for name, frames in video_buffers.items():
        if not frames: continue
        
        codec = full_meta.get("codec", "libx265")

        # Determine the file extension based on format/codec
        extension = video_format
        if video_format == 'mp4' and codec == 'ffv1':
            extension = 'mkv'

        output_path = os.path.join(output_dir, f"{name}_dynamic.{extension}")

        if video_format == 'mp4':
            mp4_pixel_format = full_meta.get("mp4_pixel_format", "yuv420p")
            print(f"  > Encoding {name} with codec: {codec}, pixel format: {mp4_pixel_format}, container: {extension}")
            
            h, w = frames[0].shape[:2]
            pad_h, pad_w = (2 - h % 2) % 2, (2 - w % 2) % 2
            frames_to_write = [np.pad(f, ((0, pad_h), (0, pad_w), (0,0)) if f.ndim==3 else ((0, pad_h), (0, pad_w)), 'constant') for f in frames]
            
            pixelformat = mp4_pixel_format if frames[0].ndim == 3 else 'gray8'
            
            ffmpeg_params = ['-loglevel', 'quiet']
            if codec in ['libx265', 'libx264']:
                ffmpeg_params, x265_opts = ['-loglevel', 'quiet'], ['log-level=none']
                if crf == 0: x265_opts.append('lossless=1')
                else: ffmpeg_params.extend(['-crf', str(crf)])
                ffmpeg_params.extend(['-x265-params', ':'.join(x265_opts)])
            
            imageio.mimwrite(
                output_path, frames_to_write, codec=codec,
                ffmpeg_params=ffmpeg_params,
                pixelformat=pixelformat, macro_block_size=1
            )
        elif video_format == 'webp':
            imageio.mimwrite(output_path, frames, format='WEBP', lossless=True)

        if os.path.exists(output_path):
            total_bytes += os.path.getsize(output_path)

    full_meta["video_format"] = video_format
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)
    total_bytes += os.path.getsize(meta_path)
    
    print(f"Total compressed size: {total_bytes / 1e6:.2f} MB")

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
    video_format = full_meta.get("video_format", "mp4")
    codec = full_meta.get("codec", "libx265")

    if total_frames > 0:
        print(f"  Loading video streams (format: {video_format}, codec: {codec})...")
        # Infer parameter names from the first frame's metadata
        param_names = set(full_meta["frames_meta"]["0"].keys()) | {"activity_mask"}
        
        extension = video_format
        if video_format == 'mp4' and codec == 'ffv1':
            extension = 'mkv'

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
    parser.add_argument("--crf", type=int, default=0, help="CRF for video compression. Lower is higher quality. 0 is lossless for some codecs.")
    parser.add_argument("--device", default="cuda:0", help="Device to use.")
    parser.add_argument("--video_format", type=str, default='mp4', choices=['mp4', 'webp'], help="Format for saving dynamic data streams.")
    parser.add_argument("--mp4_pixel_format", type=str, default='rgb24', choices=['yuv420p', 'yuv444p', 'rgb24', 'gbrp'], help="Pixel format for MP4 encoding.")
    parser.add_argument("--codec", type=str, default='libx265', choices=['libx265', 'ffv1', 'libx264'], help="Video codec for MP4 encoding.")
    args = parser.parse_args()

    if args.mode == 'compress':
        if not all([args.checkpoints_dir, args.config_path]):
            parser.error("--checkpoints_dir and --config_path are required for compression mode.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        device = args.device
        
        gop_checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "ckpt_*.pt")))
        if not gop_checkpoints:
            gop_checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pt"))) # Fallback for different naming
        
        if not gop_checkpoints:
            raise FileNotFoundError(f"No GOP checkpoints found in {args.checkpoints_dir}")

        print("--- Initializing Runner from Config ---")
        with open(args.config_path, 'r') as f:
            # Use safe_load, assuming the config is a simple structure
            trainer_config_dict = yaml.safe_load(f)

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
            for frame_in_gop in range(gop_size):
                time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
                gaussians, activity_mask = get_gaussians_for_frame(runner, time_val)
                gop_raw_frames.append({"gaussians": gaussians, "activity_mask": activity_mask})
            
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

        # --- Pass 2: Unify Frame Size and Encode Video ---
        print("\n--- Pass 2: Unifying frame sizes and preparing video buffers ---")
        
        if not processed_gops_data:
            print("No data processed. Exiting.")
            return
            
        max_sidelen = max(gop['sidelen'] for gop in processed_gops_data)
        print(f"Global video resolution will be {max_sidelen}x{max_sidelen}")

        video_buffers = defaultdict(list)
        final_frames_meta = {}
        
        # Determine all possible buffer names from the first processed frame
        all_possible_buffer_names = set(processed_gops_data[0]['frames'][0]['arrays'].keys())
        # Manually add separated quat buffers if the combined one exists
        if "quats" in all_possible_buffer_names:
            all_possible_buffer_names.remove("quats")
            for i in ['x', 'y', 'z', 'w']: all_possible_buffer_names.add(f"quats_{i}")


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
            "video_format": args.video_format,
            "codec": args.codec,
            "mp4_pixel_format": args.mp4_pixel_format,
        }
        write_output(args.output_dir, args.crf, video_buffers, full_meta)

    elif args.mode == 'decompress':
        decompress_and_export(args.output_dir, args.device)

if __name__ == "__main__":
    main()