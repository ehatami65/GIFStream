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
import sys

# Add the project root to the Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Correctly import the trainer and the specified Compression class
from long_sequence_trainer import Runner, Config, quaternion_to_rotation_matrix
from gsplat.compression.compression import Compression 
from gsplat.exporter import rgb2sh, export_splats

@torch.no_grad()
def get_neural_gaussians_for_frame(runner: Runner, time_val: float, anchor_mask: torch.Tensor):
    """
    Computes and returns the neural Gaussians for a specific time frame for a given subset of anchors.
    This function is adapted from the export_ply_sequence and get_neural_gaussians
    methods in the trainer script to generate all Gaussians for a frame.
    """
    cfg = runner.cfg
    device = runner.device
    
    assert anchor_mask is not None and anchor_mask.any(), "anchor_mask must be provided and non-empty."
    
    camtoworlds = torch.eye(4, device=device).unsqueeze(0)
    camera_ids = torch.tensor([0], device=device) if cfg.app_opt else None

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
    
    # Filter out primitives with non-positive opacity before further processing
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
    
    # Apply the selection mask to all primitive-level attributes
    opacities = neural_opacity[neural_selection_mask].squeeze(-1)
    colors = neural_colors[neural_selection_mask]
    scale_rot = neural_scale_rot[neural_selection_mask]
    offsets = transformed_offsets[neural_selection_mask]
    scales_rep = scales_repeated[neural_selection_mask]
    anchors_rep = anchors_repeated[neural_selection_mask]

    scales = scales_rep[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    quats = F.normalize(scale_rot[:, 3:7])
    means = anchors_rep + offsets
    
    sh0 = rgb2sh(colors).unsqueeze(1)
    shN = torch.zeros((sh0.shape[0], 15, 3), device=device) # Assuming max sh_degree 3

    return {
        "means": means, "scales": torch.log(scales.clamp(min=1e-8)), "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN,
    }

def load_gop_checkpoint(runner: Runner, ckpt_path: str):
    """Loads a GOP checkpoint into an existing runner instance."""
    print(f"Loading GOP checkpoint: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    
    runner.splats.load_state_dict(ckpt["splats"])
    runner.decoders.load_state_dict(ckpt["decoders"])
    if runner.cfg.app_opt and "app_module" in ckpt:
        runner.app_module.load_state_dict(ckpt["app_module"])
    
    return ckpt["active_mask"], ckpt["static_slots"], ckpt["dynamic_slots"]

def write_output(output_dir: str, crf: int, static_compressed_arrays: dict, video_buffers: dict, full_meta: dict):
    """Writes compressed static images, video buffers, and metadata to files."""
    print("\n--- Writing output files ---")
    total_bytes = 0

    # Write static data (can be multiple files per parameter)
    for name, array in static_compressed_arrays.items():
        filepath = os.path.join(output_dir, f"{name}_static.webp")
        imageio.imwrite(filepath, array, format='WEBP')
        total_bytes += os.path.getsize(filepath)

    # Write dynamic data as videos
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
    return total_bytes

def decompress_and_export(output_dir: str, device: str):
    """Decompresses the static and dynamic data and exports a full PLY sequence."""
    print("\n--- Decompressing and Exporting to PLY Sequence ---")
    meta_path = os.path.join(output_dir, "meta_bundle.json")
    if not os.path.exists(meta_path):
        print(f"Error: Meta bundle not found at {meta_path}")
        return

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
            # Check for multi-file params like means_l, means_u
            keys_to_load = [key for key in ["", "_l", "_u", "_centroids", "_labels"] if os.path.exists(os.path.join(output_dir, f"{param_name}{key}_static.webp"))]
            for key_suffix in keys_to_load:
                full_key = f"{param_name}{key_suffix}"
                static_compressed_arrays[full_key] = imageio.imread(os.path.join(output_dir, f"{full_key}_static.webp"))
        
        static_splats = compressor.decompress(full_meta["static_meta"], static_compressed_arrays, device=device)

    # 2. Load dynamic video streams
    dynamic_video_data = {}
    total_frames = full_meta.get("total_frames", 0)
    if total_frames > 0 and "dynamic_meta" in full_meta:
        print("Loading dynamic video streams...")
        param_names = full_meta["dynamic_meta"]["frames"]["0"].keys()
        for param_name in param_names:
            keys_to_load = [key for key in ["", "_l", "_u", "_centroids", "_labels"] if os.path.exists(os.path.join(output_dir, f"{param_name}{key}_dynamic.mp4"))]
            for key_suffix in keys_to_load:
                full_key = f"{param_name}{key_suffix}"
                dynamic_video_data[full_key] = imageio.mimread(os.path.join(output_dir, f"{full_key}_dynamic.mp4"), memtest=False)

    # 3. Decompress and combine frame-by-frame
    for frame_idx in tqdm(range(total_frames), desc="Exporting PLY frames"):
        dynamic_splats = {}
        if dynamic_video_data:
            frame_meta = full_meta["dynamic_meta"]["frames"][str(frame_idx)]
            frame_compressed_arrays = {name: frames[frame_idx] for name, frames in dynamic_video_data.items()}
            dynamic_splats = compressor.decompress(frame_meta, frame_compressed_arrays, device=device)

        combined_splats = {}
        all_keys = set(static_splats.keys()) | set(dynamic_splats.keys())
        
        for key in all_keys:
            static_tensor, dynamic_tensor = static_splats.get(key), dynamic_splats.get(key)
            if static_tensor is not None and dynamic_tensor is not None:
                combined_splats[key] = torch.cat([static_tensor, dynamic_tensor], dim=0)
            else:
                combined_splats[key] = static_tensor if static_tensor is not None else dynamic_tensor

        if combined_splats:
            output_ply_path = os.path.join(export_dir, f"frame_{frame_idx:05d}.ply")
            export_splats(save_to=output_ply_path, **combined_splats)

    print(f"\nDecompression complete. PLY sequence saved to: {export_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress a trained GIFStream model sequence.")
    parser.add_argument("--mode", required=True, choices=['compress', 'decompress'], help="Operation mode.")
    parser.add_argument("--gop_ckpts_dir", help="Path to the directory containing GOP checkpoints (for compression).")
    parser.add_argument("--config_path", help="Path to the original config.yml file (for compression).")
    parser.add_argument("--output_dir", default="./gop_compression_output", help="Directory for compressed/decompressed files.")
    parser.add_argument("--crf", type=int, default=0, help="CRF for video compression.")
    parser.add_argument("--device", default="cuda:0", help="Device to use.")
    parser.add_argument("--resort_interval", type=int, default=0, help="Frame interval to re-sort dynamic Gaussians. 0 to disable re-sorting after the first frame.")
    args = parser.parse_args()

    if args.mode == 'compress':
        if not all([args.gop_ckpts_dir, args.config_path]):
            parser.error("--gop_ckpts_dir and --config_path are required for compression.")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        gop_checkpoints = sorted(glob.glob(os.path.join(args.gop_ckpts_dir, "gop_*_final.pt")))
        if not gop_checkpoints:
            raise FileNotFoundError(f"No GOP checkpoints found in {args.gop_ckpts_dir}")

        print("--- Initializing Runner from GOP 0 ---")
        with open(args.config_path, 'r') as f:
            trainer_config_dict = yaml.unsafe_load(f)
        cfg = Config()
        for key, value in trainer_config_dict.items():
            if hasattr(cfg, key): setattr(cfg, key, value)
        cfg.disable_viewer = True
        
        runner = Runner(0, 0, 1, cfg)
        active_mask, static_slots, dynamic_slots = load_gop_checkpoint(runner, gop_checkpoints[0])
        static_anchor_mask = active_mask & static_slots
        dynamic_anchor_mask = active_mask & dynamic_slots

        # --- 2. Compress Static Part (ONCE) ---
        compressor = Compression(use_sort=True, verbose=False)
        if static_anchor_mask.any():
            static_meta, static_compressed_arrays, _, _ = compressor.compress(
                get_neural_gaussians_for_frame(runner, 0.0, static_anchor_mask)
            )
        else:
            static_meta = {}
            static_compressed_arrays = {}
        
        # --- 3. Compress Dynamic Part (Loop over ALL GOPs) ---
        video_buffers = defaultdict(list)
        dynamic_frames_meta = {}
        total_frames_processed = 0
        sort_indices, shn_codebook = None, None

        for gop_idx, ckpt_path in enumerate(gop_checkpoints):
            print(f"\nProcessing GOP {gop_idx} for dynamic compression...")
            load_gop_checkpoint(runner, ckpt_path)
            gop_size = runner.cfg.GOP_size
            
            for frame_in_gop in tqdm(range(gop_size), desc=f"GOP {gop_idx} Frames"):
                global_frame_idx = gop_idx * gop_size + frame_in_gop
                if global_frame_idx >= runner.cfg.total_frames: break

                if not dynamic_anchor_mask.any():
                    total_frames_processed += 1
                    continue

                time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
                dynamic_gaussians = get_neural_gaussians_for_frame(runner, time_val, anchor_mask=dynamic_anchor_mask)
                
                force_resort = args.resort_interval > 0 and total_frames_processed > 0 and total_frames_processed % args.resort_interval == 0
                
                frame_meta, frame_compressed_arrays, new_indices, new_shn_codebook = compressor.compress(
                    dynamic_gaussians,
                    sort_indices=sort_indices,
                    force_resort=force_resort,
                    shn_initial_centroids=None if force_resort else shn_codebook
                )

                if new_indices is not None: sort_indices = new_indices
                if new_shn_codebook is not None: shn_codebook = new_shn_codebook

                for name, array in frame_compressed_arrays.items():
                    video_buffers[name].append(array)
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