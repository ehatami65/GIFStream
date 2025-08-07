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
import shutil
import sys
from gsplat.exporter import rgb2sh
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_trainer_GIFStream import Runner, Config
from gsplat.compression import Compression
from gsplat.exporter import export_splats
from simple_trainer_GIFStream import quaternion_to_rotation_matrix
from utils import find_k_neighbors

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
    camera_ids = torch.tensor([0], device=device) if cfg.app_opt else None

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

    neural_selection_mask = (neural_opacity < 0.0).view(-1)
    neural_opacity[neural_selection_mask] = -1e10

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
    selected_opacity = neural_opacity.squeeze(-1)
    selected_colors = neural_colors
    selected_scale_rot = neural_scale_rot
    selected_offsets = transformed_offsets
    scales_repeated = scales_repeated
    anchors_repeated = anchors_repeated

    scales = scales_repeated[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
    quats = F.normalize(selected_scale_rot[:, 3:7])
    means = anchors_repeated + selected_offsets
    opacities = selected_opacity
    
    # Clamp scales to avoid issues with log(0)
    log_scales = torch.log(scales.clamp(min=1e-8))

    colors = selected_colors
    sh0 = rgb2sh(colors).unsqueeze(1)
    # For now, we only handle degree 0 SH. This must match export_ply_sequence.
    shN = torch.zeros((sh0.shape[0], 0, 3), device=device)
    # --- End of adapted get_neural_gaussians ---

    return {
        "means": means,
        "scales": log_scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
        "neural_opacity": neural_opacity,
    }


def plot_factors_histogram(runner: Runner, output_dir: str):
    """Plots and saves a histogram of the model's factors."""
    print("\n--- Plotting factors histogram ---")

    if runner.cfg.compression_sim:
        # When compression_sim is on, factors are already quantized/mapped to [0, 1]
        # and stored in comp_sim_splats during model loading.
        print("Plotting factors from `comp_sim_splats` (quantized for simulation).")
        factors_to_plot = runner.comp_sim_splats["factors"].detach().cpu().numpy()
    else:
        # When compression_sim is off, factors are raw logits.
        # We apply sigmoid to visualize them in a meaningful [0, 1] range.
        print("Plotting factors from `splats` (raw logits), applying sigmoid for visualization.")
        factors_to_plot = torch.sigmoid(runner.splats["factors"]).detach().cpu().numpy()

    factor_names = ["Time Feature", "Motion", "KNN", "Pruning"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution of Factors', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if i < factors_to_plot.shape[1]:
            ax.hist(factors_to_plot[:, i], bins=200, alpha=0.7, range=(0, 1))
            ax.set_title(f'Factor {i}: {factor_names[i]}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "factors_histogram.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Factors histogram saved to {save_path}")


def load_gifstream_model(config_path: str, ckpt_path: str, device: str) -> Runner:
    """Loads the GIFStream runner and checkpoint."""
    print("--- Loading model and configuration ---")
    with open(config_path, 'r') as f:
        trainer_config_dict = yaml.unsafe_load(f)

    cfg = Config()
    for key, value in trainer_config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    cfg.disable_viewer = True
    
    runner = Runner(0, 0, 1, cfg)
    
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    splats_dict = ckpt.get("splats", {})
    for k in runner.splats.keys():
        if k in splats_dict:
            runner.splats[k].data = splats_dict[k].to(device)
    
    if "decoders" in ckpt:
        runner.decoders.load_state_dict(ckpt["decoders"])
    runner.decoders.to(device)

    # Handle compression simulation attributes from the checkpoint
    runner.cfg.compression_sim = ckpt.get("compression_sim", runner.cfg.compression_sim)
    if runner.cfg.compression_sim:
        print("Compression simulation is enabled. Loading entropy models...")
        runner.load_entropy_model_from_ckpt(ckpt, cfg.entropy_model_type)

    # Also load knn if it was used during training
    if cfg.knn:
        _, runner.indices = find_k_neighbors(runner.splats["anchors"], cfg.n_knn)

    print("Model loaded successfully.")
    return runner


def compress_scene(runner: Runner, resort_interval: int, static_threshold: float):
    """
    Separates scene into static and dynamic anchors, then compresses them.
    - Static anchors are compressed into a single set of PNG images.
    - Dynamic anchors are compressed into a video stream.
    """
    print("\n--- Separating and Compressing Scene ---")
    device = runner.device
    compressor = Compression(use_sort=True, verbose=False, seed=42)
    total_compression_time = 0.0

    # 1. Identify static and dynamic anchors
    print(f"Identifying static anchors with threshold {static_threshold}...")
    if runner.cfg.compression_sim:
        factors = runner.comp_sim_splats["factors"]
    else:
        factors = runner.splats["factors"]

    time_factor = factors[:, 0]
    static_mask = torch.abs(time_factor) < static_threshold
    dynamic_mask = ~static_mask

    static_indices = torch.where(static_mask)[0]
    dynamic_indices = torch.where(dynamic_mask)[0]

    print(f"Found {len(static_indices)} static anchors and {len(dynamic_indices)} dynamic anchors.")

    # 2. Process and compress static data (one-time operation)
    static_images = {}
    static_meta = {}
    if len(static_indices) > 0:
        print("Compressing static data...")
        # Generate Gaussians for static anchors at time=0
        static_gaussians_full = get_neural_gaussians_for_frame(runner, time_val=0, anchor_mask=static_mask)
        
        # Apply opacity culling to keep only visible static Gaussians
        neural_opacity = static_gaussians_full.pop("neural_opacity")
        neural_selection_mask = (neural_opacity > 0.0).view(-1)
        
        static_gaussians_culled = {}
        for key, tensor in static_gaussians_full.items():
            if tensor is not None and tensor.shape[0] > 0:
                static_gaussians_culled[key] = tensor[neural_selection_mask]
        
        # Compress the final culled static Gaussians into PNG images
        t0_compress_static = time.time()
        static_meta, static_images, _, _ = compressor.compress(
            static_gaussians_culled, force_resort=True
        )
        total_compression_time += time.time() - t0_compress_static

    # 3. Process and compress dynamic data (frame-by-frame video)
    print("Compressing dynamic data into video...")
    video_buffers = {
        "means_l": [], "means_u": [], "scales": [], "sh0": [],
        "shN_centroids": [], "shN_labels": [], "opacities": [],
        "quats_x": [], "quats_y": [], "quats_z": [], "quats_w": [],
    }
    dynamic_frames_meta = {}
    sort_indices = None
    shn_codebook = None
    keep_indices = None

    if len(dynamic_indices) > 0:
        gop_size = runner.cfg.GOP_size
        for frame_idx in tqdm(range(gop_size), desc="Compressing dynamic frames"):
            time_val = frame_idx / (gop_size - 1) if gop_size > 1 else 0

            # Generate Gaussians for dynamic anchors at the current time
            splats_dict = get_neural_gaussians_for_frame(runner, time_val, anchor_mask=dynamic_mask)
            splats_dict.pop("neural_opacity") # Not needed for dynamic part

            force_resort = resort_interval > 0 and frame_idx > 0 and frame_idx % resort_interval == 0
            if force_resort:
                keep_indices = None

            n_gaussians = splats_dict["means"].shape[0]
            if n_gaussians > 0:
                n_sidelen = int(np.sqrt(n_gaussians))
                n_crop = n_gaussians - n_sidelen**2

                if n_crop > 0 and keep_indices is None:
                    print(f"\nFrame {frame_idx}: Cropping {n_crop} dynamic Gaussians based on opacity.")
                    keep_indices = torch.argsort(splats_dict["opacities"], descending=True)[:-n_crop]

                if keep_indices is not None:
                    for key, tensor in splats_dict.items():
                        if tensor is not None and tensor.shape[0] == n_gaussians:
                            splats_dict[key] = tensor[keep_indices]
            
            t0_compress_dynamic = time.time()
            meta, compressed_arrays, new_indices, new_shn_codebook = compressor.compress(
                splats_dict,
                sort_indices=sort_indices,
                force_resort=force_resort,
                shn_initial_centroids=None if force_resort else shn_codebook,
            )
            total_compression_time += time.time() - t0_compress_dynamic

            if new_indices is not None: sort_indices = new_indices
            if new_shn_codebook is not None: shn_codebook = new_shn_codebook

            for name, array in compressed_arrays.items():
                if name == "quats":
                    video_buffers["quats_x"].append(array[..., 0])
                    video_buffers["quats_y"].append(array[..., 1])
                    video_buffers["quats_z"].append(array[..., 2])
                    video_buffers["quats_w"].append(array[..., 3])
                elif name in video_buffers:
                    video_buffers[name].append(array)
            
            dynamic_frames_meta[str(frame_idx)] = meta

    # 4. Assemble final metadata
    full_meta = {
        "scene_info": {"total_anchors": len(factors)},
        "static_scene": {
            "meta": static_meta,
            "indices": static_indices.cpu().numpy().tolist(),
        },
        "dynamic_scene": {
            "meta": {"frames": dynamic_frames_meta},
            "indices": dynamic_indices.cpu().numpy().tolist(),
        }
    }

    return static_images, video_buffers, full_meta, total_compression_time


def write_output(output_dir: str, crf: int, static_images: dict, video_buffers: dict, full_meta: dict):
    """Writes compressed static images, video buffers, and metadata to files."""
    print("\n--- Writing output files ---")
    total_compressed_size_mb = 0.0

    # Write static images (PNG)
    if static_images:
        print("Writing static PNG files...")
        static_mapping = {}
        for name, array in tqdm(static_images.items(), desc="Writing static files"):
            is_grayscale = array.ndim == 2
            ext = ".png"
            filename = f"{name}_static{ext}"
            image_path = os.path.join(output_dir, filename)
            
            # imageio v2 API automatically handles mode based on array shape.
            # The 'mode' argument for the pillow plugin caused a TypeError.
            imageio.imwrite(image_path, array)
            total_compressed_size_mb += os.path.getsize(image_path) / 1e6
            
            storage_info = {"width": array.shape[1], "height": array.shape[0], "file": filename}
            if is_grayscale:
                storage_info["type"] = "image_grayscale"
            else:
                storage_info["type"] = "image_rgb" if array.shape[2] == 3 else "image_rgba"
            
            static_mapping[name] = storage_info
        
        full_meta["static_scene"]["file_mapping"] = static_mapping

    # Write dynamic videos (MP4)
    if any(video_buffers.values()):
        print("Writing dynamic video files...")
        dynamic_mapping = {}
        for name, frames in tqdm(video_buffers.items(), desc="Writing video files"):
            if not frames:
                continue
            
            video_path = os.path.join(output_dir, f"{name}.mp4")
            
            h, w = frames[0].shape[:2]
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2

            frames_to_write = [np.pad(f, ((0, pad_h), (0, pad_w)) if f.ndim==2 else ((0, pad_h), (0, pad_w), (0,0)), 'constant') for f in frames] if pad_h > 0 or pad_w > 0 else frames
            is_grayscale = frames_to_write[0].ndim == 2
            
            ffmpeg_params, x265_opts = ['-loglevel', 'quiet'], ['log-level=none']
            if crf == 0: x265_opts.append('lossless=1')
            else: ffmpeg_params.extend(['-crf', str(crf)])
            ffmpeg_params.extend(['-x265-params', ':'.join(x265_opts)])
            
            imageio.mimwrite(
                video_path, frames_to_write, codec='libx265',
                ffmpeg_params=ffmpeg_params, pixelformat='gray' if is_grayscale else 'gbrp', macro_block_size=1
            )
            
            total_compressed_size_mb += os.path.getsize(video_path) / 1e6

            storage_info = {"width": w, "height": h}
            param = "quats" if name.startswith("quats_") else name
            if param == "quats":
                if param not in dynamic_mapping:
                    dynamic_mapping[param] = {"type": "video_split_channel", "files": {}}
                dynamic_mapping[param]["files"][name] = f"{name}.mp4"
                dynamic_mapping[param].update(storage_info)
            else:
                storage_info.update({"type": "video_grayscale" if is_grayscale else "video_rgb", "file": f"{name}.mp4"})
                dynamic_mapping[param] = storage_info
        
        full_meta["dynamic_scene"]["file_mapping"] = dynamic_mapping

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=4)
    total_compressed_size_mb += os.path.getsize(meta_path) / 1e6
    return total_compressed_size_mb


def decompress_scene_to_ply(output_dir: str, device: str):
    """
    Decompresses data into separate static and dynamic .ply files.
    - A single `static.ply` is created for all static Gaussians.
    - A sequence of `frame_dynamic_xxxx.ply` files is created for the dynamic Gaussians.
    """
    print("\n--- Decompressing files to PLY sequence ---")
    meta_path = os.path.join(output_dir, "meta.json")
    decompressed_dir = os.path.join(output_dir, "decompressed_plys")
    os.makedirs(decompressed_dir, exist_ok=True)

    with open(meta_path, "r") as f:
        full_meta = json.load(f)

    compressor = Compression(use_sort=False, verbose=False)
    total_decompression_time = 0.0

    # 1. Decompress and save static data (once)
    if "static_scene" in full_meta and full_meta["static_scene"]["meta"]:
        print("Loading and decompressing static data...")
        static_meta = full_meta["static_scene"]["meta"]
        static_mapping = full_meta["static_scene"]["file_mapping"]
        static_images_loaded = {}
        for param_name, mapping in static_mapping.items():
            image_path = os.path.join(output_dir, mapping["file"])
            static_images_loaded[param_name] = imageio.imread(image_path)
        
        t0_decompress_static = time.time()
        static_splats = compressor.decompress(static_meta, static_images_loaded, device=device)
        total_decompression_time += time.time() - t0_decompress_static

        static_ply_path = os.path.join(decompressed_dir, "static.ply")
        export_splats(
            save_to=static_ply_path,
            means=static_splats.get("means"),
            scales=static_splats.get("scales"),
            quats=static_splats.get("quats"),
            opacities=static_splats.get("opacities"),
            sh0=static_splats.get("sh0"),
            shN=static_splats.get("shN"),
        )
        print(f"  Saved static Gaussians to: {static_ply_path}")

    # 2. Decompress dynamic data (frame by frame)
    dynamic_video_data = {}
    num_frames = 0
    if "dynamic_scene" in full_meta and full_meta["dynamic_scene"]["meta"]["frames"]:
        num_frames = len(full_meta["dynamic_scene"]["meta"]["frames"])
        print("Loading dynamic video files into memory...")
        dynamic_mapping = full_meta["dynamic_scene"]["file_mapping"]
        for mapping in tqdm(dynamic_mapping.values(), desc="Loading videos"):
            files_to_load = []
            if mapping['type'] == 'video_split_channel':
                files_to_load.extend(mapping['files'].values())
            else:
                files_to_load.append(mapping.get('file'))

            for file in files_to_load:
                if file and file not in dynamic_video_data:
                    video_path = os.path.join(output_dir, file)
                    dynamic_video_data[file] = imageio.mimread(video_path)

    # 3. Export dynamic PLY sequence
    if num_frames > 0:
        for frame_idx in tqdm(range(num_frames), desc="Decompressing dynamic frames"):
            dynamic_meta = full_meta["dynamic_scene"]["meta"]["frames"][str(frame_idx)]
            dynamic_mapping = full_meta["dynamic_scene"]["file_mapping"]
            compressed_arrays_loaded = {}

            for param_name, mapping in dynamic_mapping.items():
                storage_type = mapping["type"]
                base_param = param_name.split('_')[0]

                if base_param not in dynamic_meta:
                    continue

                if storage_type == "video_split_channel":
                    files_dict = mapping["files"]
                    ordered_keys = [f"quats_{c}" for c in ['x', 'y', 'z', 'w']]
                    channel_frames = [dynamic_video_data[files_dict[key]][frame_idx] for key in ordered_keys if key in files_dict]

                    orig_h, orig_w = mapping['height'], mapping['width']
                    cropped_channels = [cf[:orig_h, :orig_w] for cf in channel_frames]

                    if cropped_channels and cropped_channels[0].ndim == 3:
                        cropped_channels = [c[..., 0] for c in cropped_channels]

                    reconstructed_array = np.stack(cropped_channels, axis=-1)
                    compressed_arrays_loaded[param_name] = reconstructed_array
                
                elif storage_type in ["video_rgb", "video_grayscale"]:
                    file = mapping["file"]
                    if file in dynamic_video_data:
                        frame_data = dynamic_video_data[file][frame_idx]
                        orig_h, orig_w = mapping['height'], mapping['width']
                        compressed_arrays_loaded[param_name] = frame_data[:orig_h, :orig_w]

            t0_decompress_dynamic = time.time()
            dynamic_splats = compressor.decompress(dynamic_meta, compressed_arrays_loaded, device=device)
            total_decompression_time += time.time() - t0_decompress_dynamic

            output_ply_path = os.path.join(decompressed_dir, f"frame_dynamic_{frame_idx:04d}.ply")
            export_splats(
                save_to=output_ply_path,
                means=dynamic_splats.get("means"),
                scales=dynamic_splats.get("scales"),
                quats=dynamic_splats.get("quats"),
                opacities=dynamic_splats.get("opacities"),
                sh0=dynamic_splats.get("sh0"),
                shN=dynamic_splats.get("shN"),
            )
    
    avg_decompression_time = total_decompression_time / num_frames if num_frames > 0 else 0
    print("\n--- Decompression Summary ---")
    print(f"  Decompressed {num_frames} dynamic frames.")
    print(f"  Decompressed files saved to: {decompressed_dir}")
    print(f"  Average decompression time: {avg_decompression_time:.2f} s/frame")


def print_compression_summary(gop_size, total_compressed_size_mb, total_compression_time, output_dir):
    """Prints a summary of the compression process."""
    avg_compression_time = total_compression_time / gop_size if gop_size > 0 else 0
    print("\n--- Compression Summary ---")
    print(f"  Processed {gop_size} frames from the model.")
    print(f"  Total compressed size: {total_compressed_size_mb:.2f} MB")
    print(f"  Average compression time: {avg_compression_time:.2f} s/frame")
    print(f"  Compressed files saved to: {output_dir}")
    print("\nCompression complete!")


def main():
    """Main function to run the scene compression."""
    parser = argparse.ArgumentParser(description="Compress a trained GIFStream model into videos.")
    parser.add_argument("--config_path", required=True, help="Path to the config.yml file.")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (.pt).")
    parser.add_argument("--output_dir", default="./scene_compression_output", help="Directory for compressed files.")
    parser.add_argument(
        "--crf", type=int, default=0,
        help="Constant Rate Factor for video compression. 0 for lossless, higher for smaller files."
    )
    parser.add_argument(
        "--resort_interval",
        type=int,
        default=0,
        help="Interval for re-sorting frames to adapt to changes. 0 to disable.",
    )
    parser.add_argument(
        "--static_threshold",
        type=float,
        default=-1,
        help="Threshold for considering an anchor as static based on its time factor.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to use (e.g., cuda:0).")
    parser.add_argument("--skip_histogram", action="store_true", help="Skip plotting histogram.")
    parser.add_argument("--skip_decompress", action="store_true", help="Skip decompressing to PLY.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the model
    runner = load_gifstream_model(args.config_path, args.ckpt, args.device)
    
    # (Optional) Plot histograms for debugging
    if not args.skip_histogram:
        plot_factors_histogram(runner, args.output_dir)
    
    # 2. Separate scene and compress static/dynamic components
    static_images, video_buffers, full_meta, total_compression_time = compress_scene(
        runner, args.resort_interval, args.static_threshold
    )

    # 3. Write compressed data to files
    total_compressed_size_mb = write_output(
        args.output_dir, args.crf, static_images, video_buffers, full_meta
    )

    # 4. Print summary
    print_compression_summary(runner.cfg.GOP_size, total_compressed_size_mb, total_compression_time, args.output_dir)

    # 5. (Optional) Decompress the files back to .ply files
    if not args.skip_decompress:
        decompress_scene_to_ply(args.output_dir, args.device)


if __name__ == "__main__":
    main()
