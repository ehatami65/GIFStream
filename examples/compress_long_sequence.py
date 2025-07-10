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
def get_neural_gaussians_for_frame(runner: Runner, time_val: float, anchor_mask: torch.Tensor = None):
    """
    Computes and returns the neural Gaussians for a specific time frame.
    This function is adapted from the export_ply_sequence and get_neural_gaussians
    methods in simple_trainer_GIFStream.py to generate all Gaussians for a frame.
    """
    cfg = runner.cfg
    device = runner.device

    # If no mask is provided, we use all anchors. Otherwise, use the provided mask.
    if anchor_mask is None:
        visible_anchor_mask = torch.ones(runner.splats["anchors"].shape[0], dtype=torch.bool, device=device)
    else:
        visible_anchor_mask = anchor_mask
    
    camtoworlds = torch.eye(4, device=device).unsqueeze(0)
    camera_ids = torch.tensor([-1], device=device) if cfg.app_opt else None

    # This logic is adapted from `get_neural_gaussians` and `decoding_features`
    if not cfg.compression_sim or not hasattr(runner, 'comp_sim_splats'):
        selected_anchors = runner.splats["anchors"][visible_anchor_mask]
        selected_offsets = runner.splats["offsets"][visible_anchor_mask]
    else:
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

    neural_selection_mask = (neural_opacity > 0.0).view(-1)

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

    # Apply mask to select valid Gaussians
    selected_opacity = neural_opacity[neural_selection_mask].squeeze(-1)
    selected_colors = neural_colors[neural_selection_mask]
    selected_scale_rot = neural_scale_rot[neural_selection_mask]
    selected_offsets = transformed_offsets[neural_selection_mask]
    scales_repeated = scales_repeated[neural_selection_mask]
    anchors_repeated = anchors_repeated[neural_selection_mask]

    scales = scales_repeated[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
    quats = F.normalize(selected_scale_rot[:, 3:7])
    means = anchors_repeated + selected_offsets
    opacities = selected_opacity
    
    # Clamp scales to avoid issues with log(0)
    log_scales = torch.log(scales.clamp(min=1e-8))

    colors = selected_colors
    sh0 = rgb2sh(colors).unsqueeze(1)

    return {
        "means": means,
        "scales": log_scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
    }
    
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
        "means": means, "scales": torch.log(scales.clamp(min=1e-8)), "quats": quats,
        "opacities": opacities, "sh0": sh0,
    }, primitive_activity_mask


def write_output(output_dir: str, crf: int, static_compressed_arrays: dict, video_buffers: dict, full_meta: dict):
    """Writes compressed static images, video buffers, and metadata to files."""
    print("\n--- Writing output files ---")
    total_bytes = 0

    for name, array in static_compressed_arrays.items():
        filepath = os.path.join(output_dir, f"{name}_static.webp")
        imageio.imwrite(filepath, array, format='WEBP', lossless=True)
        total_bytes += os.path.getsize(filepath)

    for name, frames in video_buffers.items():
        if not frames: continue
        video_path = os.path.join(output_dir, f"{name}_dynamic.mp4")
        
        h, w = frames[0].shape[:2]
        pad_h, pad_w = (2 - h % 2) % 2, (2 - w % 2) % 2
        frames_to_write = [np.pad(f, ((0, pad_h), (0, pad_w), (0,0)) if f.ndim==3 else ((0, pad_h), (0, pad_w)), 'constant') for f in frames]

        pixelformat = 'gray8' if frames[0].ndim == 2 else 'yuv420p'
        imageio.mimwrite(
            video_path, frames_to_write, codec='libx265',
            ffmpeg_params=['-loglevel', 'quiet', '-crf', str(crf)],
            pixelformat=pixelformat, macro_block_size=1
        )
        total_bytes += os.path.getsize(video_path)

    meta_path = os.path.join(output_dir, "meta_bundle.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=2)
    total_bytes += os.path.getsize(meta_path)
    
    print(f"Total compressed size: {total_bytes / 1e6:.2f} MB")

def decompress_and_export(output_dir: str, device: str):
    """
    Decompresses the static and dynamic data and exports a full PLY sequence.
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

    # 1. Decompress static data
    static_splats = {}
    if "static_meta" in full_meta and full_meta["static_meta"]:
        print("Decompressing static data...")
        static_compressed_arrays = {}
        for param_name in full_meta["static_meta"].keys():
            if param_name == "quats":
                channels = []
                for i in range(4):
                    filepath = os.path.join(output_dir, f"quats_{i}_static.webp")
                    img = imageio.imread(filepath)
                    channels.append(img[..., 0] if img.ndim == 3 else img)
                static_compressed_arrays["quats"] = np.stack(channels, axis=-1)
            elif os.path.exists(os.path.join(output_dir, f"{param_name}_static.webp")):
                 static_compressed_arrays[param_name] = imageio.imread(os.path.join(output_dir, f"{param_name}_static.webp"))
            elif os.path.exists(os.path.join(output_dir, f"{param_name}_l_static.webp")):
                static_compressed_arrays[f"{param_name}_l"] = imageio.imread(os.path.join(output_dir, f"{param_name}_l_static.webp"))
                static_compressed_arrays[f"{param_name}_u"] = imageio.imread(os.path.join(output_dir, f"{param_name}_u_static.webp"))
        
        static_splats = compressor.decompress(full_meta["static_meta"], static_compressed_arrays, device=device)

    # 2. Load dynamic video streams
    dynamic_video_data = defaultdict(list)
    total_frames = full_meta.get("total_frames", 0)
    if total_frames > 0 and "dynamic_meta" in full_meta and full_meta["dynamic_meta"]["frames"]:
        print("Loading dynamic video streams...")
        param_names = set(full_meta["dynamic_meta"]["frames"]["0"].keys()) | {"activity_mask"}
        for param_name in param_names:
            if param_name == "quats":
                for i in range(4):
                    video_path = os.path.join(output_dir, f"quats_{i}_dynamic.mp4")
                    if os.path.exists(video_path): dynamic_video_data[f"quats_{i}"] = imageio.mimread(video_path, memtest=False)
            else:
                for suffix in ["", "_l", "_u"]:
                     video_path = os.path.join(output_dir, f"{param_name}{suffix}_dynamic.mp4")
                     if os.path.exists(video_path): dynamic_video_data[f"{param_name}{suffix}"] = imageio.mimread(video_path, memtest=False)
        if os.path.exists(os.path.join(output_dir, "activity_mask_dynamic.mp4")):
            dynamic_video_data["activity_mask"] = imageio.mimread(os.path.join(output_dir, "activity_mask_dynamic.mp4"), memtest=False)

    # 3. Decompress and combine frame-by-frame
    for frame_idx in tqdm(range(total_frames), desc="Exporting PLY frames"):
        dynamic_splats = {}
        if dynamic_video_data:
            frame_meta = full_meta["dynamic_meta"]["frames"].get(str(frame_idx))
            if not frame_meta: continue

            frame_compressed_arrays = {}
            for param_name in frame_meta.keys():
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

            dynamic_splats = compressor.decompress(frame_meta, frame_compressed_arrays, device=device)
            
            if "activity_mask" in dynamic_video_data:
                # --- THIS IS THE FIX ---
                activity_frame_raw = dynamic_video_data["activity_mask"][frame_idx]
                # Ensure the mask is single-channel before flattening
                activity_frame_gray = activity_frame_raw[..., 0] if activity_frame_raw.ndim == 3 else activity_frame_raw
                activity_mask_frame = activity_frame_gray.flatten() > 128
                # --- END FIX ---
                
                for key, tensor in dynamic_splats.items():
                    if tensor is not None:
                        dynamic_splats[key] = tensor[activity_mask_frame]

        combined_splats = {k: torch.cat([static_splats.get(k), v], dim=0) for k, v in dynamic_splats.items() if static_splats.get(k) is not None}
        if not combined_splats and static_splats: combined_splats = static_splats
        elif not combined_splats and dynamic_splats: combined_splats = dynamic_splats

        if combined_splats:
            output_ply_path = os.path.join(export_dir, f"frame_{frame_idx:05d}.ply")
            combined_splats.pop("shN", None)
            export_splats(save_to=output_ply_path, **combined_splats)

    print(f"\nDecompression complete. PLY sequence saved to: {export_dir}")

def export_pre_compression_plys(runner, output_dir, static_anchor_mask, changeable_anchor_mask, gop_checkpoints):
    """
    Generates and saves the full sequence of PLY files BEFORE any compression
    to serve as a ground truth for debugging.
    """
    print("\n--- Exporting Pre-Compression PLY Sequence ---")
    export_dir = os.path.join(output_dir, "pre_compression_plys")
    os.makedirs(export_dir, exist_ok=True)
    device = runner.device

    # 1. Generate and save static data (once)
    static_splats = {}
    if static_anchor_mask.any():
        print("Generating static Gaussians...")
        # Use the video frame function to get the activity mask for culling
        static_gaussians_full, static_activity_mask = get_neural_gaussians_for_video_frame(runner, 0.0, static_anchor_mask)
        
        static_splats = {}
        for key, tensor in static_gaussians_full.items():
            if tensor is not None:
                static_splats[key] = tensor[static_activity_mask]

        if static_splats:
            static_ply_path = os.path.join(export_dir, "static.ply")
            export_splats(save_to=static_ply_path, **static_splats)
            print(f"  Saved static Gaussians to: {static_ply_path}")

    # 2. Generate and save dynamic data (frame-by-frame)
    total_frames_processed = 0

    for gop_idx, ckpt_path in enumerate(gop_checkpoints):
        print(f"\nProcessing GOP {gop_idx} for pre-compression export...")
        runner.load_gop_checkpoint(ckpt_path)
        gop_size = runner.cfg.GOP_size
        
        for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
            global_frame_idx = gop_idx * gop_size + frame_in_gop
            if global_frame_idx >= runner.cfg.total_frames: break

            dynamic_splats = {}
            if changeable_anchor_mask.any():
                time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
                dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(runner, time_val, anchor_mask=changeable_anchor_mask)
                
                # Filter by activity mask
                for key, tensor in dynamic_gaussians.items():
                    if tensor is not None:
                        dynamic_splats[key] = tensor[primitive_activity_mask]
                        

            # Save the dynamic-only PLY
            if dynamic_splats:
                dynamic_ply_path = os.path.join(export_dir, f"dynamic_frame_{global_frame_idx:05d}.ply")
                export_splats(save_to=dynamic_ply_path, **dynamic_splats)

            
            total_frames_processed += 1

    print(f"\nPre-compression export complete. PLY sequence saved to: {export_dir}")

def test_lossless_cycle(runner: Runner, output_dir: str, static_anchor_mask, changeable_anchor_mask, gop_checkpoints):
    """
    Performs an in-memory compression/decompression cycle to test the quantization
    and data handling logic, bypassing the lossy video codec step.
    """
    print("\n--- Testing Lossless Quantization/Dequantization Cycle ---")
    export_dir = os.path.join(output_dir, "lossless_cycle_plys")
    os.makedirs(export_dir, exist_ok=True)
    device = runner.device

    # --- 1. Compress & Decompress Static Part (ONCE) ---
    static_splats = {}
    if static_anchor_mask.any():
        print("Processing static part...")
        static_compressor = Compression(use_sort=True, verbose=False)
        static_gaussians = get_neural_gaussians_for_frame(runner, 0.0, static_anchor_mask)
        
        # "Compress" to get metadata and numpy arrays
        static_meta, static_numpy_arrays, _, _ = static_compressor.compress(static_gaussians, force_resort=True)
        
        # "Decompress" immediately from the numpy arrays
        static_splats = static_compressor.decompress(static_meta, static_numpy_arrays, device=device)

    # --- 2. Process Dynamic Part (Loop over ALL GOPs) ---
    total_frames_processed = 0
    sort_indices, shn_codebook = None, None
    
    # We need to calculate padding info once
    num_changeable_primitives = changeable_anchor_mask.sum().item() * runner.cfg.n_offsets
    n_sidelen_video = math.ceil(math.sqrt(num_changeable_primitives))
    padded_size = n_sidelen_video**2
    num_padding = padded_size - num_changeable_primitives
    
    dynamic_compressor = Compression(use_sort=True, verbose=False)

    for gop_idx, ckpt_path in enumerate(gop_checkpoints):
        print(f"\nProcessing GOP {gop_idx} for lossless cycle test...")
        runner.load_gop_checkpoint(ckpt_path)
        gop_size = runner.cfg.GOP_size
        
        for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
            global_frame_idx = gop_idx * gop_size + frame_in_gop
            if global_frame_idx >= runner.cfg.total_frames: break

            if not changeable_anchor_mask.any():
                total_frames_processed += 1
                continue

            time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
            dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(runner, time_val, anchor_mask=changeable_anchor_mask)
            
            # --- START LOSSLESS CYCLE FOR THIS FRAME ---
            # Pad the data
            padded_gaussians = {}
            for name, tensor in dynamic_gaussians.items():
                pad_shape = list(tensor.shape); pad_shape[0] = num_padding
                padding = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
                padded_gaussians[name] = torch.cat([tensor, padding], dim=0)
            
            # "Compress"
            force_resort = (total_frames_processed == 0) # Only sort on the very first frame
            frame_meta, frame_numpy_arrays, new_indices, _ = dynamic_compressor.compress(
                padded_gaussians, sort_indices=sort_indices, force_resort=force_resort
            )
            if new_indices is not None: sort_indices = new_indices

            # "Decompress"
            decompressed_padded_splats = dynamic_compressor.decompress(frame_meta, frame_numpy_arrays, device=device)

            # Un-pad using the activity mask
            # We need to re-sort the activity mask just like the data was
            padded_activity_mask = torch.cat([primitive_activity_mask, torch.zeros(num_padding, dtype=torch.bool, device=device)], dim=0)
            sorted_activity_mask = padded_activity_mask[sort_indices]
            
            dynamic_splats = {}
            for key, tensor in decompressed_padded_splats.items():
                if tensor is not None:
                    dynamic_splats[key] = tensor[sorted_activity_mask]
            # --- END LOSSLESS CYCLE ---

            # --- Combine and save the resulting PLY ---
            combined_splats = {}
            # ... (your combination logic) ...
            all_keys = set(static_splats.keys()) | set(dynamic_splats.keys())
            for key in all_keys:
                s_tensor = static_splats.get(key)
                d_tensor = dynamic_splats.get(key)
                if s_tensor is not None and d_tensor is not None and d_tensor.shape[0] > 0:
                    combined_splats[key] = torch.cat([s_tensor, d_tensor], dim=0)
                elif s_tensor is not None: combined_splats[key] = s_tensor
                elif d_tensor is not None: combined_splats[key] = d_tensor

            if combined_splats.get("means") is not None:
                output_ply_path = os.path.join(export_dir, f"frame_lossless_cycle_{global_frame_idx:05d}.ply")
                # Ensure shN exists for the exporter
                if "shN" not in combined_splats:
                    combined_splats["shN"] = torch.zeros((combined_splats["means"].shape[0], 0, 3), device=device)
                export_splats(save_to=output_ply_path, **combined_splats)
            
            total_frames_processed += 1

    print(f"\nLossless cycle test complete. PLY sequence saved to: {export_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress a trained GIFStream model sequence.")
    parser.add_argument("--mode", required=True, choices=['compress', 'decompress', 'export_pre_compression', 'test_lossless_cycle'],
                         help="Operation mode.")
    parser.add_argument("--gop_ckpts_dir", help="Path to the directory containing GOP checkpoints (for compression).")
    parser.add_argument("--config_path", help="Path to the original config.yml file (for compression).")
    parser.add_argument("--output_dir", default="./gop_compression_output", help="Directory for compressed/decompressed files.")
    parser.add_argument("--crf", type=int, default=0, help="CRF for video compression.")
    parser.add_argument("--device", default="cuda:0", help="Device to use.")
    parser.add_argument("--resort_interval", type=int, default=0, help="Frame interval to re-sort dynamic Gaussians. 0 to disable re-sorting after the first frame.")
    args = parser.parse_args()

    if args.mode in ['compress', 'export_pre_compression', 'test_lossless_cycle']:
        if not all([args.gop_ckpts_dir, args.config_path]):
            parser.error("--gop_ckpts_dir and --config_path are required for compression.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        device = args.device
        
        gop_checkpoints = sorted(glob.glob(os.path.join(args.gop_ckpts_dir, "gop_*_final.pt")))
        if not gop_checkpoints:
            raise FileNotFoundError(f"No GOP checkpoints found in {args.gop_ckpts_dir}")

        print("--- Initializing Runner and Scene Partition from GOP 0 ---")
        with open(args.config_path, 'r') as f:
            trainer_config_dict = yaml.unsafe_load(f)

        cfg = Config()
        for key, value in trainer_config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        
        cfg.disable_viewer = True

        runner = Runner(0, 0, 1, cfg)
        runner.load_gop_checkpoint(gop_checkpoints[0])
        static_anchor_mask = runner.active_mask & runner.static_slots
        changeable_anchor_mask = runner.dynamic_slots | ~runner.active_mask

        if args.mode == 'export_pre_compression':
            export_pre_compression_plys(runner, args.output_dir, static_anchor_mask, changeable_anchor_mask, gop_checkpoints)
            return

        if args.mode == 'test_lossless_cycle':
            test_lossless_cycle(runner, args.output_dir, static_anchor_mask, changeable_anchor_mask, gop_checkpoints)
            return

        # --- 2. Compress Static Part (ONCE) ---
        static_meta, static_compressed_arrays = {}, {}
        if static_anchor_mask.any():
            static_compressor = Compression(use_sort=True, verbose=False)
            static_gaussians = get_neural_gaussians_for_frame(runner, 0.0, static_anchor_mask)
            static_meta, static_compressed_arrays, _, _ = static_compressor.compress(static_gaussians, force_resort=True)

        # --- 3. Compress Dynamic Part (Loop over ALL GOPs) ---
        video_buffers = defaultdict(list)
        dynamic_frames_meta = {}
        total_frames_processed = 0
        sort_indices, shn_codebook = None, None
        
        num_changeable_primitives = changeable_anchor_mask.sum().item() * cfg.n_offsets
        n_sidelen_video = math.ceil(math.sqrt(num_changeable_primitives))
        padded_size = n_sidelen_video**2
        num_padding = padded_size - num_changeable_primitives

        dynamic_compressor = Compression(use_sort=True, verbose=False)

        for gop_idx, ckpt_path in enumerate(gop_checkpoints):
            print(f"\nProcessing GOP {gop_idx} for dynamic compression...")
            runner.load_gop_checkpoint(ckpt_path)
            gop_size = runner.cfg.GOP_size
            force_resort = True
            for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
                global_frame_idx = gop_idx * gop_size + frame_in_gop
                if global_frame_idx >= runner.cfg.total_frames: break

                if not changeable_anchor_mask.any():
                    total_frames_processed += 1
                    continue

                time_val = frame_in_gop / (gop_size - 1)
                dynamic_gaussians, primitive_activity_mask = get_neural_gaussians_for_video_frame(runner, time_val, anchor_mask=changeable_anchor_mask)
                
                # Pad all tensors before compression
                padded_gaussians = {}
                for name, tensor in dynamic_gaussians.items():
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = num_padding
                    padding = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
                    padded_gaussians[name] = torch.cat([tensor, padding], dim=0)
                
                padded_activity_mask = torch.cat([primitive_activity_mask, torch.zeros(num_padding, dtype=torch.bool, device=device)], dim=0)

                

                frame_meta, frame_compressed_arrays, new_indices, new_shn_codebook = dynamic_compressor.compress(
                    padded_gaussians,
                    sort_indices=sort_indices,
                    force_resort=force_resort,
                    shn_initial_centroids=None if force_resort else shn_codebook
                )
                # force_resort = (args.resort_interval > 0 and total_frames_processed > 0 and total_frames_processed % args.resort_interval == 0) or (total_frames_processed == 0)
                force_resort = False
                if new_indices is not None: sort_indices = new_indices
                
                if "quats" in frame_compressed_arrays:
                    quat_array = frame_compressed_arrays.pop("quats")
                    for i in range(4):
                        video_buffers[f"quats_{i}"].append(quat_array[..., i])

                for name, array in frame_compressed_arrays.items():
                    video_buffers[name].append(array)
                
                # Sort and buffer the activity mask
                sorted_activity_mask = padded_activity_mask[sort_indices].cpu().numpy()
                activity_grid = sorted_activity_mask.reshape(n_sidelen_video, n_sidelen_video)
                video_buffers["activity_mask"].append((activity_grid * 255).astype(np.uint8))

                dynamic_frames_meta[str(global_frame_idx)] = frame_meta
                total_frames_processed += 1


        # --- 4. Write Output ---
        full_meta = {
            "static_meta": static_meta,
            "dynamic_meta": {"frames": dynamic_frames_meta},
            "total_frames": total_frames_processed,
        }
        write_output(args.output_dir, args.crf, static_compressed_arrays, video_buffers, full_meta)

    elif args.mode == 'decompress':
        decompress_and_export(args.output_dir, args.device)

if __name__ == "__main__":
    main()