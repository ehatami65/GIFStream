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

from long_sequence_trainer import Runner, Config, quaternion_to_rotation_matrix
from gsplat.compression.compression import Compression
from gsplat.exporter import rgb2sh, export_splats

@torch.no_grad()
def get_neural_gaussians_for_video_frame(runner: Runner, time_val: float, anchor_mask: torch.Tensor):
    """
    Computes neural Gaussians for a specific time frame for a given subset of anchors.
    Also returns a primitive-level activity mask.
    """
    cfg = runner.cfg
    device = runner.device
    
    assert anchor_mask is not None and anchor_mask.any(), "anchor_mask must be provided and non-empty."
    
    camtoworlds = torch.eye(4, device=device).unsqueeze(0)
    camera_ids = torch.tensor([-1], device=device) if cfg.app_opt else None

    splats_to_use = runner.splats
    selected_anchors = splats_to_use["anchors"][anchor_mask]
    selected_offsets = splats_to_use["offsets"][anchor_mask]

    results = runner.decoding_features(
        camtoworlds, time_val, anchor_mask, canonical=False, step=-1, camera_ids=camera_ids
    )
    
    neural_opacity = results["neural_opacity"]
    neural_colors = results["neural_colors"]
    neural_scale_rot = results["neural_scale_rot"]
    motion = results["motion"]
    selected_scales = results["selected_scales"]

    # This is the key mask: True for any primitive that should be rendered.
    is_anchor_active = runner.active_mask[anchor_mask]
    primitive_parent_is_active_mask = is_anchor_active.repeat_interleave(cfg.n_offsets)
    primitive_activity_mask = (neural_opacity > 0.0).view(-1) & primitive_parent_is_active_mask
    
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

    return {
        "means": means, "scales": torch.log(scales.clamp(min=1e-10)), "quats": quats,
        "opacities": opacities, "sh0": sh0,
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
                ffmpeg_params.extend(['-crf', str(crf)])
            
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
    """
    print("\n--- Decompressing and Exporting to PLY Sequence ---")
    
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta bundle not found at {meta_path}")

    with open(meta_path, "r") as f:
        full_meta = json.load(f)

    export_dir = os.path.join(output_dir, "decompressed_ply_sequence")
    os.makedirs(export_dir, exist_ok=True)
    
    compressor = Compression(use_sort=False)

    # 1. Load the unified dynamic video streams
    dynamic_video_data = defaultdict(list)
    total_frames = full_meta.get("total_frames", 0)
    video_format = full_meta.get("video_format", "mp4")
    codec = full_meta.get("codec", "libx265")

    if total_frames > 0:
        print(f"  Loading dynamic data streams (format: {video_format}, codec: {codec})...")
        # Infer parameter names from the first frame's metadata
        param_names = set(full_meta["frames_meta"]["0"].keys()) | {"activity_mask"}
        
        extension = video_format
        if video_format == 'mp4' and codec == 'ffv1':
            extension = 'mkv'

        for param_name in param_names:
            if param_name == "quats":
                for i in range(4):
                    video_path = os.path.join(output_dir, f"quats_{i}_dynamic.{extension}")
                    if os.path.exists(video_path): dynamic_video_data[f"quats_{i}"] = imageio.mimread(video_path, memtest=False)
            else:
                for suffix in ["", "_l", "_u"]:
                    video_path = os.path.join(output_dir, f"{param_name}{suffix}_dynamic.{extension}")
                    if os.path.exists(video_path): dynamic_video_data[f"{param_name}{suffix}"] = imageio.mimread(video_path, memtest=False)
            
            activity_path = os.path.join(output_dir, f"activity_mask_dynamic.{extension}")
            if os.path.exists(activity_path) and "activity_mask" not in dynamic_video_data:
                dynamic_video_data["activity_mask"] = imageio.mimread(activity_path, memtest=False)

    # 2. Decompress and combine frame-by-frame
    for frame_idx in tqdm(range(total_frames), desc="Exporting PLY frames"):
        dynamic_splats = {}
        if dynamic_video_data:
            frame_meta = full_meta["frames_meta"].get(str(frame_idx))
            if not frame_meta: continue
            
            frame_compressed_arrays = {}
            for param_name in frame_meta.keys():
                 if param_name == "gop_idx": continue
                 if param_name == "quats":
                    quat_channels = []
                    for i in range(4):
                        frame = dynamic_video_data[f"quats_{i}"][frame_idx]
                        quat_channels.append(frame[..., 0] if frame.ndim == 3 else frame)
                    frame_compressed_arrays["quats"] = np.stack(quat_channels, axis=-1)
                 elif f"{param_name}_l" in dynamic_video_data:
                    frame_compressed_arrays[f"{param_name}_l"] = dynamic_video_data[f"{param_name}_l"][frame_idx]
                    frame_compressed_arrays[f"{param_name}_u"] = dynamic_video_data[f"{param_name}_u"][frame_idx]
                 elif param_name in dynamic_video_data:
                    frame_compressed_arrays[param_name] = dynamic_video_data[param_name][frame_idx]

            n_sidelen_video = full_meta.get("n_sidelen_video")
            if n_sidelen_video:
                for key, array in frame_compressed_arrays.items():
                    if array.ndim >= 2:
                        h, w = array.shape[:2]
                        if h > n_sidelen_video or w > n_sidelen_video:
                            frame_compressed_arrays[key] = array[:n_sidelen_video, :n_sidelen_video]

            for key, array in frame_compressed_arrays.items():
                if array.ndim == 3 and array.shape[-1] == 4:
                    base_param_name = key.split('_')[0]
                    if base_param_name in frame_meta:
                        meta_shape = frame_meta[base_param_name].get("shape")
                        if meta_shape and len(meta_shape) > 1 and meta_shape[-1] == 3:
                            frame_compressed_arrays[key] = array[..., :3]
            
            dynamic_splats_padded = compressor.decompress(frame_meta, frame_compressed_arrays, device=device)
            
            activity_mask_padded = None
            if "activity_mask" in dynamic_video_data:
                activity_frame_raw = dynamic_video_data["activity_mask"][frame_idx]
                if n_sidelen_video:
                    h, w = activity_frame_raw.shape[:2]
                    if h > n_sidelen_video or w > n_sidelen_video:
                        activity_frame_raw = activity_frame_raw[:n_sidelen_video, :n_sidelen_video]
                activity_frame_gray = activity_frame_raw[..., 0] if activity_frame_raw.ndim == 3 else activity_frame_raw
                activity_mask_padded = (activity_frame_gray.flatten() > 128)
            
            # Un-pad based on this frame's actual primitive count
            num_primitives_in_frame = frame_meta.get("means", {}).get("shape", [0])[0]
            
            dynamic_splats = {
                name: tensor[:num_primitives_in_frame] 
                for name, tensor in dynamic_splats_padded.items() if tensor is not None
            }
            
            if activity_mask_padded is not None:
                activity_mask = torch.tensor(activity_mask_padded[:num_primitives_in_frame], device=device)
                
                for key, tensor in dynamic_splats.items():
                    if tensor is not None:
                        dynamic_splats[key] = tensor[activity_mask]

            if dynamic_splats.get("means") is not None and dynamic_splats["means"].shape[0] > 0:
                output_ply_path = os.path.join(export_dir, f"frame_{frame_idx:05d}.ply")
                dynamic_splats.pop("shN", None)
                export_splats(save_to=output_ply_path, **dynamic_splats)

    print(f"\nDecompression complete. PLY sequence saved to: {export_dir}")

def export_pre_compression_plys(runner, output_dir, gop_checkpoints):
    """
    Generates and saves the full sequence of PLY files BEFORE any compression
    to serve as a ground truth for debugging, operating on a per-GOP basis.
    """
    print("\n--- Exporting Pre-Compression PLY Sequence ---")
    export_dir = os.path.join(output_dir, "pre_compression_plys")
    os.makedirs(export_dir, exist_ok=True)
    
    for gop_idx, ckpt_path in enumerate(gop_checkpoints):
        print(f"\nProcessing GOP {gop_idx} for pre-compression export...")
        runner.load_gop_checkpoint(ckpt_path)
        gop_size = runner.cfg.GOP_size
        active_anchor_mask = runner.active_mask
        
        if not active_anchor_mask.any():
            print(f"  > No active anchors in GOP {gop_idx}, skipping.")
            continue
            
        for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
            global_frame_idx = gop_idx * gop_size + frame_in_gop
            if global_frame_idx >= runner.cfg.total_frames: break

            time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
            dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(runner, time_val, anchor_mask=active_anchor_mask)
            
            dynamic_splats = {}
            for key, tensor in dynamic_gaussians.items():
                if tensor is not None:
                    dynamic_splats[key] = tensor[primitive_activity_mask]
            
            if dynamic_splats:
                ply_path = os.path.join(export_dir, f"frame_{global_frame_idx:05d}.ply")
                export_splats(save_to=ply_path, **dynamic_splats)

    print(f"\nPre-compression export complete. PLY sequence saved to: {export_dir}")

def test_lossless_cycle(runner: Runner, output_dir: str, gop_checkpoints):
    """
    Performs an in-memory compression/decompression cycle on a per-GOP basis.
    """
    print("\n--- Testing Lossless Quantization/Dequantization Cycle (Per-GOP) ---")
    export_dir = os.path.join(output_dir, "lossless_cycle_plys")
    os.makedirs(export_dir, exist_ok=True)
    device = runner.device

    for gop_idx, ckpt_path in enumerate(gop_checkpoints):
        print(f"\nProcessing GOP {gop_idx} for lossless cycle test...")
        runner.load_gop_checkpoint(ckpt_path)
        gop_size = runner.cfg.GOP_size
        
        active_anchor_mask = runner.active_mask
        if not active_anchor_mask.any():
            print(f"  > No active anchors in GOP {gop_idx}, skipping.")
            continue

        num_active_primitives = active_anchor_mask.sum().item() * runner.cfg.n_offsets
        n_sidelen_video = math.ceil(math.sqrt(num_active_primitives))
        padded_size = n_sidelen_video**2
        num_padding = padded_size - num_active_primitives
        
        dynamic_compressor = Compression(use_sort=True, verbose=False)
        sort_indices = None
        force_resort = True

        for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
            global_frame_idx = gop_idx * gop_size + frame_in_gop
            if global_frame_idx >= runner.cfg.total_frames: break

            time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
            dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(
                runner, time_val, anchor_mask=active_anchor_mask
            )
            
            # --- START LOSSLESS CYCLE FOR THIS FRAME ---
            padded_gaussians = {}
            for name, tensor in dynamic_gaussians.items():
                pad_shape = list(tensor.shape); pad_shape[0] = num_padding
                padding = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
                padded_gaussians[name] = torch.cat([tensor, padding], dim=0)
            
            frame_meta, frame_numpy_arrays, new_indices, _ = dynamic_compressor.compress(
                padded_gaussians, sort_indices=sort_indices, force_resort=force_resort
            )
            force_resort = False # Only sort on the first frame of the GOP
            if new_indices is not None: sort_indices = new_indices

            decompressed_padded_splats = dynamic_compressor.decompress(frame_meta, frame_numpy_arrays, device=device)

            padded_activity_mask = torch.cat([primitive_activity_mask, torch.zeros(num_padding, dtype=torch.bool, device=device)], dim=0)
            sorted_activity_mask = padded_activity_mask[sort_indices]
            
            dynamic_splats = {}
            for key, tensor in decompressed_padded_splats.items():
                if tensor is not None:
                    dynamic_splats[key] = tensor[sorted_activity_mask]
            # --- END LOSSLESS CYCLE ---

            if dynamic_splats.get("means") is not None:
                output_ply_path = os.path.join(export_dir, f"frame_lossless_cycle_{global_frame_idx:05d}.ply")
                if "shN" not in dynamic_splats:
                    dynamic_splats["shN"] = torch.zeros((dynamic_splats["means"].shape[0], 0, 3), device=device)
                export_splats(save_to=output_ply_path, **dynamic_splats)

    print(f"\nLossless cycle test complete. PLY sequence saved to: {export_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress a trained GIFStream model sequence.")
    parser.add_argument("--mode", required=True, choices=['compress', 'decompress', 'export_pre_compression', 'test_lossless_cycle'],
                         help="Operation mode.")
    parser.add_argument("--gop_ckpts_dir", help="Path to the directory containing GOP checkpoints (for compression modes).")
    parser.add_argument("--config_path", help="Path to the original config.yml file (for compression modes).")
    parser.add_argument("--output_dir", default="./gop_compression_output", help="Directory for compressed/decompressed files.")
    parser.add_argument("--crf", type=int, default=0, help="CRF for video compression.")
    parser.add_argument("--device", default="cuda:0", help="Device to use.")
    parser.add_argument("--video_format", type=str, default='webp', choices=['mp4', 'webp'], help="Format for saving dynamic data streams.")
    parser.add_argument("--mp4_pixel_format", type=str, default='rgb24', choices=['yuv420p', 'yuv444p', 'rgb24', 'gbrp'], help="Pixel format for MP4 encoding.")
    parser.add_argument("--codec", type=str, default='libx265', choices=['libx265', 'ffv1', 'libx264'], help="Video codec for MP4 encoding.")
    args = parser.parse_args()

    if args.mode in ['compress', 'export_pre_compression', 'test_lossless_cycle']:
        if not all([args.gop_ckpts_dir, args.config_path]):
            parser.error("--gop_ckpts_dir and --config_path are required for this mode.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        device = args.device
        
        gop_checkpoints = sorted(glob.glob(os.path.join(args.gop_ckpts_dir, "gop_*_final.pt")))
        if not gop_checkpoints:
            raise FileNotFoundError(f"No GOP checkpoints found in {args.gop_ckpts_dir}")

        print("--- Initializing Runner from Config ---")
        with open(args.config_path, 'r') as f:
            trainer_config_dict = yaml.unsafe_load(f)

        cfg = Config()
        for key, value in trainer_config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        cfg.disable_viewer = True
        runner = Runner(0, 0, 1, cfg)

        if args.mode == 'export_pre_compression':
            export_pre_compression_plys(runner, args.output_dir, gop_checkpoints)
            return

        if args.mode == 'test_lossless_cycle':
            test_lossless_cycle(runner, args.output_dir, gop_checkpoints)
            return

        # --- COMPRESSION MODE ---
        
        # --- Pass 1: Cache all frame data by running the model once per frame ---
        print("--- Pass 1: Caching all frame data (running model once per frame) ---")
        cached_frames = []
        for gop_idx, ckpt_path in enumerate(tqdm(gop_checkpoints, desc="GOP Caching Pass")):
            runner.load_gop_checkpoint(ckpt_path)
            active_anchor_mask = runner.active_mask
            gop_size = runner.cfg.GOP_size

            if not active_anchor_mask.any():
                for _ in range(gop_size):
                    if len(cached_frames) >= runner.cfg.total_frames: break
                    cached_frames.append({"empty": True, "gop_idx": gop_idx})
                continue
            
            for frame_in_gop in range(gop_size):
                if len(cached_frames) >= runner.cfg.total_frames: break
                
                time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
                dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(
                    runner, time_val, anchor_mask=active_anchor_mask
                )
                cached_frames.append({
                    "gaussians": dynamic_gaussians,
                    "activity_mask": primitive_activity_mask,
                    "gop_idx": gop_idx,
                    "empty": False
                })
        
        # --- Pass 2: Analyze cache, compress, and write to video ---
        print("--- Pass 2: Analyzing cache and compressing all frames ---")

        # --- Sub-pass 2a: Analyze GOPs from cache to find GOP-wide masks ---
        gops_analysis = {}
        for frame_data in cached_frames:
            if frame_data["empty"]: continue
            gop_idx = frame_data["gop_idx"]
            if gop_idx not in gops_analysis: gops_analysis[gop_idx] = []
            gops_analysis[gop_idx].append(frame_data["activity_mask"])
        
        gops_meta_info = {}
        max_gop_active_primitives = 0
        for gop_idx, masks in gops_analysis.items():
            gop_wide_activity_mask = torch.stack(masks).any(dim=0)
            num_gop_active_primitives = gop_wide_activity_mask.sum().item()
            gops_meta_info[gop_idx] = {
                "activity_mask": gop_wide_activity_mask,
                "num_active": num_gop_active_primitives
            }
            if num_gop_active_primitives > max_gop_active_primitives:
                max_gop_active_primitives = num_gop_active_primitives
        
        if max_gop_active_primitives == 0:
            print("No active primitives found in any frame. No output will be generated.")
            return

        print(f"  > Max active primitives in any single GOP: {max_gop_active_primitives}")
        
        # --- Sub-pass 2b: Process each GOP to create locally-squared, sorted frame data ---
        all_gops_frame_data = []
        max_sidelen = 0
        dynamic_compressor = Compression(use_sort=True, verbose=False)

        for gop_idx in range(len(gop_checkpoints)):
            gop_info = gops_meta_info.get(gop_idx, {"num_active": 0})
            num_gop_active_primitives = gop_info["num_active"]

            gop_frames_output = {
                "arrays": defaultdict(list),
                "meta": [],
                "sidelen": 0
            }

            if num_gop_active_primitives > 0:
                gop_wide_mask = gop_info["activity_mask"]
                n_sidelen_gop = math.ceil(math.sqrt(num_gop_active_primitives))
                if n_sidelen_gop > max_sidelen: max_sidelen = n_sidelen_gop
                
                num_padding = (n_sidelen_gop**2) - num_gop_active_primitives
                gop_frames_output["sidelen"] = n_sidelen_gop
                sort_indices = None
                force_resort = True

                gop_cached_frames = [f for f in cached_frames if f["gop_idx"] == gop_idx]

                for frame_data in gop_cached_frames:
                    if frame_data["empty"]:
                        gop_frames_output["meta"].append({})
                        continue

                    filtered_gaussians = {name: tensor[gop_wide_mask] for name, tensor in frame_data["gaussians"].items()}
                    
                    # Pad the data to a perfect square *before* compression and sorting
                    padded_gaussians = {}
                    for name, tensor in filtered_gaussians.items():
                        pad_shape = list(tensor.shape); pad_shape[0] = num_padding
                        padding = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
                        padded_gaussians[name] = torch.cat([tensor, padding], dim=0)

                    frame_meta, frame_compressed_arrays, new_indices, _ = dynamic_compressor.compress(
                        padded_gaussians, sort_indices=sort_indices, force_resort=force_resort
                    )
                    if new_indices is not None: sort_indices = new_indices
                    force_resort = False
                    
                    # Also pad the activity mask and apply the same sorting
                    current_frame_activity_mask = frame_data["activity_mask"][gop_wide_mask]
                    padded_activity_mask = torch.cat([current_frame_activity_mask, torch.zeros(num_padding, dtype=torch.bool, device=device)], dim=0)
                    sorted_activity_mask = padded_activity_mask[sort_indices].cpu().numpy()
                    activity_grid = sorted_activity_mask.reshape(n_sidelen_gop, n_sidelen_gop)
                    frame_compressed_arrays["activity_mask"] = (activity_grid * 255).astype(np.uint8)

                    for name, array in frame_compressed_arrays.items():
                        gop_frames_output["arrays"][name].append(array)
                    gop_frames_output["meta"].append(frame_meta)
            
            all_gops_frame_data.append(gop_frames_output)

        # --- Sub-pass 2c: Pad all GOP frames to the max uniform size and buffer for writing ---
        print(f"  > Padding all frames to uniform size ({max_sidelen}x{max_sidelen}) and buffering...")
        video_buffers = defaultdict(list)
        final_frames_meta = {}
        global_frame_idx_counter = 0

        # Define all possible buffer names that might be created during compression
        all_possible_buffer_names = set()
        for gop_frames_output in all_gops_frame_data:
            all_possible_buffer_names.update(gop_frames_output["arrays"].keys())
        # Manually add separated quat buffers if the combined one exists
        if "quats" in all_possible_buffer_names:
            all_possible_buffer_names.remove("quats")
            for i in range(4): all_possible_buffer_names.add(f"quats_{i}")

        for gop_output in all_gops_frame_data:
            sidelen = gop_output["sidelen"]
            pad_h = pad_w = max_sidelen - sidelen
            num_frames_in_gop = len(gop_output["meta"])

            gop_arrays = gop_output["arrays"]

            for i in range(num_frames_in_gop):
                # Handle all possible buffers to ensure synchronization
                for name in all_possible_buffer_names:
                    if name in gop_arrays or (name.startswith("quats_") and "quats" in gop_arrays):
                        if name.startswith("quats_"):
                            quat_idx = int(name.split('_')[1])
                            frame = gop_arrays["quats"][i][..., quat_idx]
                        else:
                            frame = gop_arrays[name][i]

                        padding = ((0, pad_h), (0, pad_w), (0,0)) if frame.ndim == 3 else ((0, pad_h), (0, pad_w))
                        padded_frame = np.pad(frame, padding, 'constant')
                        video_buffers[name].append(padded_frame)
                    else:
                        # Append blank frames for buffers not present in this GOP
                        shape = (max_sidelen, max_sidelen)
                        dtype = np.uint8
                        if name == "means":
                            shape += (3,)
                            dtype = np.uint16
                        elif name not in ['activity_mask', 'labels'] and not name.startswith('quats_'):
                            shape += (3,)
                        
                        video_buffers[name].append(np.zeros(shape, dtype=dtype))
                
                # Update the metadata to reflect the final padded shape
                meta_for_frame = gop_output["meta"][i]
                if meta_for_frame:
                    padded_len = max_sidelen * max_sidelen
                    for param_meta in meta_for_frame.values():
                        if "shape" in param_meta:
                            param_meta["shape"][0] = padded_len

                final_frames_meta[str(global_frame_idx_counter)] = meta_for_frame
                global_frame_idx_counter += 1
        
        # --- Pass 3: Write unified video files and metadata ---
        print(f"--- Pass 3: Writing output files ---")
        full_meta = {
            "frames_meta": final_frames_meta,
            "n_sidelen_video": max_sidelen,
            "total_frames": len(cached_frames),
            "video_format": args.video_format,
            "codec": args.codec,
            "mp4_pixel_format": args.mp4_pixel_format,
        }
        write_output(args.output_dir, args.crf, video_buffers, full_meta)

    elif args.mode == 'decompress':
        decompress_and_export(args.output_dir, args.device)

if __name__ == "__main__":
    main()