import os
import argparse
import time
import yaml
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import json
import shutil
import sys

from gsplat import Compression
from gsplat.exporter import export_splats, import_splats, load_ply




def _encode_image(img, lossless: bool = True, quality: int = 100):
    """
    Compresses the image as webp.
    """
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.fromarray(img).save(
        buffer,
        format="webp",
        lossless=lossless,
        quality=quality if not lossless else 100,
        method=6,
        exact=True,
    )
    return buffer.getvalue()


def main():
    """Main function to run the dynamic compression."""
    parser = argparse.ArgumentParser(
        description="Compress a sequence of .ply files for dynamic scenes."
    )
    parser.add_argument("--input_dir", help="Input directory containing .ply files.")
    parser.add_argument(
        "--output_dir",
        default="./dynamic_compression_output",
        help="Directory for compressed files.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="Constant Rate Factor for video compression. 0 for lossless, higher for smaller files.",
    )
    parser.add_argument(
        "--lossy-params",
        nargs="+",
        default=[],
        help="List of parameters to compress lossily (not used in video mode).",
    )
    parser.add_argument(
        "--resort_interval",
        type=int,
        default=0,
        help="Interval for re-sorting frames to adapt to changes. 0 to disable.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Compression Step ---
    print("\n--- Compressing files into separate videos ---")

    # Dictionary to hold frames for each video stream
    video_buffers = {
        "means_l": [], "means_u": [], "scales": [], "sh0": [],
        "shN_centroids": [], "shN_labels": [], "opacities": [],
        "quats_x": [], "quats_y": [], "quats_z": [], "quats_w": [],
    }

    # Centralized metadata
    full_meta = {"file_mapping": {}, "frames": {}}

    # Initialize compressor
    compressor = Compression(use_sort=True, verbose=False, seed=42)

    ply_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".ply")])
    sort_indices = None
    shn_codebook = None
    total_compression_time = 0.0
    total_input_size_mb = 0.0
    keep_indices = None

    for frame_idx, ply_file in tqdm(enumerate(ply_files), desc="Compressing frames", total=len(ply_files)):
        input_ply_path = os.path.join(args.input_dir, ply_file)
        total_input_size_mb += os.path.getsize(input_ply_path) / 1e6
        
        force_resort = args.resort_interval > 0 and frame_idx > 0 and frame_idx % args.resort_interval == 0
        if force_resort:
            keep_indices = None
        # Load Original Splats
        means, scales, quats, opacities, sh0, shN = import_splats(input_ply_path, device=args.device)
        n_gaussians = means.shape[0]
        n_sidelen = int(np.sqrt(n_gaussians))
        n_crop = n_gaussians - n_sidelen**2
    
        
        if n_crop > 0 and keep_indices is None:
            print(f"Number of Gaussians ({n_gaussians}) is not a perfect square. Removing {n_crop} based on first frame's opacities.")
            # Sort by opacity and get indices to keep (the ones with highest opacity)
            keep_indices = torch.argsort(opacities.squeeze(), descending=True)[:-n_crop]
        else:
            means = means[keep_indices]
            scales = scales[keep_indices]
            quats = quats[keep_indices]
            opacities = opacities[keep_indices]
            sh0 = sh0[keep_indices]
            # Handle case where shN might not exist or be empty
            if shN is not None and shN.shape[0] > 0:
                shN = shN[keep_indices]

        splats_dict = {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}

        # Determine if a re-sort is needed
        t0_compress = time.time()
        

        meta, compressed_arrays, new_indices, new_shn_codebook = compressor.compress(
            splats_dict,
            sort_indices=sort_indices,
            force_resort=force_resort,
            shn_initial_centroids=None if force_resort else shn_codebook,
        )
        total_compression_time += time.time() - t0_compress

        if new_indices is not None: sort_indices = new_indices
        if new_shn_codebook is not None: shn_codebook = new_shn_codebook

        # Distribute compressed arrays into their respective video buffers
        for name, array in compressed_arrays.items():
            if name == "quats":
                video_buffers["quats_x"].append(array[..., 0])
                video_buffers["quats_y"].append(array[..., 1])
                video_buffers["quats_z"].append(array[..., 2])
                video_buffers["quats_w"].append(array[..., 3])
            elif name in video_buffers:
                video_buffers[name].append(array)

        meta["original_ply"] = ply_file
        full_meta['frames'][str(frame_idx)] = meta

    # --- Write Video Files and Finalize Metadata ---
    total_compressed_size_mb = 0.0
    for name, frames in tqdm(video_buffers.items(), desc="Writing video files"):
        if not frames:
            continue
        
        video_path = os.path.join(args.output_dir, f"{name}.mp4")
        
        h, w = frames[0].shape[:2]
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2

        if pad_h > 0 or pad_w > 0:
            padded_frames = [np.pad(f, ((0, pad_h), (0, pad_w)) if f.ndim==2 else ((0, pad_h), (0, pad_w), (0,0)), 'constant') for f in frames]
            frames_to_write = padded_frames
        else:
            frames_to_write = frames

        is_grayscale = frames_to_write[0].ndim == 2
        
        # Base ffmpeg parameters to silence logs
        ffmpeg_params = ['-loglevel', 'quiet']

        # Configure codec and parameters
        codec = 'libx265'
        x265_opts = ['log-level=none']  # Silence the x265-specific logger
        if args.crf == 0:
            x265_opts.append('lossless=1')
        else:
            # For lossy mode, -crf is the top-level ffmpeg param
            ffmpeg_params.extend(['-crf', str(args.crf)])
        
        # Pass all accumulated x265 options
        ffmpeg_params.extend(['-x265-params', ':'.join(x265_opts)])
        
        pixel_format = 'gray' if is_grayscale else 'gbrp'


        imageio.mimwrite(
            video_path,
            frames_to_write,
            codec=codec,
            ffmpeg_params=ffmpeg_params,
            pixelformat=pixel_format,
            macro_block_size=1,
        )
        
        total_compressed_size_mb += os.path.getsize(video_path) / 1e6

        # Update file mapping in metadata with original dimensions
        storage_info = {"width": w, "height": h}
        if name.startswith("quats_"):
            param = "quats"
            if param not in full_meta["file_mapping"]:
                full_meta["file_mapping"][param] = {"type": "video_split_channel", "files": {}}
            full_meta["file_mapping"][param]["files"][name] = f"{name}.mp4"
            full_meta["file_mapping"][param].update(storage_info)
        else:
            param = name
            storage_info.update({
                "type": "video_grayscale" if is_grayscale else "video_rgb",
                "file": f"{name}.mp4"
            })
            full_meta["file_mapping"][param] = storage_info


    # Save the centralized metadata file
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=4)
    total_compressed_size_mb += os.path.getsize(meta_path) / 1e6

    # --- Print Compression Summary ---
    avg_compression_time = total_compression_time / len(ply_files)
    compression_ratio = total_input_size_mb / total_compressed_size_mb if total_compressed_size_mb > 0 else float('inf')

    print("\n--- Compression Summary ---")
    print(f"  Processed {len(ply_files)} PLY files.")
    print(f"  Total input size: {total_input_size_mb:.2f} MB")
    print(f"  Total compressed size: {total_compressed_size_mb:.2f} MB")
    print(f"  Overall compression ratio: {compression_ratio:.2f}x")
    print(f"  Average compression time: {avg_compression_time:.2f} s/frame")
    print(f"  Compressed videos and metadata saved to: {args.output_dir}")

    # --- Decompression Step ---
    print("\n--- Decompressing files ---")
    decompressed_dir = os.path.join(args.output_dir, "decompressed_plys")
    os.makedirs(decompressed_dir, exist_ok=True)

    # Load the centralized metadata
    with open(meta_path, "r") as f:
        full_meta = json.load(f)

    # Read all video streams into memory
    video_data = {}
    print("Loading video files into memory...")
    for mapping in tqdm(full_meta["file_mapping"].values(), desc="Loading videos"):
        
        files_to_load = []
        if mapping['type'] == 'video_split_channel':
            files_to_load.extend(mapping['files'].values())
        else:
            files_to_load.append(mapping.get('file'))

        for file in files_to_load:
            if file and file not in video_data:
                video_path = os.path.join(args.output_dir, file)
                video_data[file] = imageio.mimread(video_path)

    total_decompression_time = 0.0
    frame_count = len(full_meta["frames"])
    
    for frame_idx in tqdm(range(frame_count), desc="Decompressing frames"):
        meta = full_meta["frames"][str(frame_idx)]
        original_ply_name = meta.pop("original_ply", f"frame_{frame_idx:04d}.ply")
        
        # Reconstruct the compressed_arrays dictionary for this frame
        compressed_arrays_loaded = {}
        
        # Iterate over the file manifest to load data for the current frame
        for param_name, mapping in full_meta["file_mapping"].items():
            storage_type = mapping["type"]
            
            # The base parameter name, e.g., 'means' from 'means_l'
            base_param = param_name.split('_')[0]

            # Only load data for parameters that are actually in the frame's meta
            if base_param not in meta:
                continue

            if storage_type == "video_split_channel":
                # This handles 'quats' which is split into 4 grayscale videos
                files_dict = mapping["files"]
                # Ensure correct x,y,z,w order, not alphabetical
                ordered_keys = [f"quats_{c}" for c in ['x', 'y', 'z', 'w']]
                channel_frames = [video_data[files_dict[key]][frame_idx] for key in ordered_keys if key in files_dict]

                # Crop frames to original size before stacking
                orig_h, orig_w = mapping['height'], mapping['width']
                cropped_channels = [cf[:orig_h, :orig_w] for cf in channel_frames]

                # If grayscale videos were read as 3-channel, take only one channel
                if cropped_channels and cropped_channels[0].ndim == 3:
                    cropped_channels = [c[..., 0] for c in cropped_channels]

                reconstructed_array = np.stack(cropped_channels, axis=-1)
                compressed_arrays_loaded[param_name] = reconstructed_array
            
            elif storage_type in ["video_rgb", "video_grayscale"]:
                # This handles all other parameters stored in their own videos.
                file = mapping["file"]
                if file in video_data:
                    frame_data = video_data[file][frame_idx]
                    # Crop frame to original size
                    orig_h, orig_w = mapping['height'], mapping['width']
                    compressed_arrays_loaded[param_name] = frame_data[:orig_h, :orig_w]

        # Decompress
        t0_decompress = time.time()
        decompressed_splats = compressor.decompress(meta, compressed_arrays_loaded, device=args.device)
        total_decompression_time += time.time() - t0_decompress

        # Ensure all keys exist for consistent export
        for key in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
            if key not in decompressed_splats:
                # This part might need adjustment based on expected shapes for empty tensors
                decompressed_splats[key] = torch.empty((0, 3), device=args.device)

        # Export the decompressed splats to a .ply file
        output_ply_path = os.path.join(decompressed_dir, original_ply_name)
        export_splats(
            save_to=output_ply_path,
            means=decompressed_splats.get("means"),
            scales=decompressed_splats.get("scales"),
            quats=decompressed_splats.get("quats"),
            opacities=decompressed_splats.get("opacities"),
            sh0=decompressed_splats.get("sh0"),
            shN=decompressed_splats.get("shN"),
        )
    
    avg_decompression_time = total_decompression_time / frame_count if frame_count > 0 else 0
    print("\n--- Decompression Summary ---")
    print(f"  Decompressed {frame_count} frames.")
    print(f"  Decompressed files saved to: {decompressed_dir}")
    print(f"  Average decompression time: {avg_decompression_time:.2f} s/frame")
    print("\nDecompression complete!")


if __name__ == "__main__":
    main() 