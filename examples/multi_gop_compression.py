import os
import glob
import re
from dataclasses import dataclass, field
from typing import List, Dict

import torch
import tyro
from tqdm import tqdm
import numpy as np
import yaml
import torch.nn.functional as F

# --- Library Imports ---
from compression.compression_helper import Compression
from compression.chunk_processor import ChunkProcessor
from compression.decompression_player import DecompressionPlayer
from compression import io_utils
from compression.io_ply import export_splats
from gsplat.exporter import rgb2sh

# --- GIFStream Specific Imports (from your legacy context) ---
# Make sure these are available in your Python path
from simple_trainer_GIFStream import Runner, Config, quaternion_to_rotation_matrix
from utils import find_k_neighbors

# --- Legacy Helper Functions (unchanged) ---

@torch.no_grad()
def get_gaussians_for_frame(runner: Runner, time_val: float, previous_gaussians: dict = None):
    """
    Computes and returns the neural Gaussians for a specific time frame.
    Also returns a primitive-level activity mask.
    If previous_gaussians is provided, values for inactive primitives are
    carried forward from previous_gaussians.
    """
    cfg = runner.cfg
    device = runner.device
    
    visible_anchor_mask = torch.ones(runner.splats["anchors"].shape[0], dtype=torch.bool, device=device)
    camtoworlds = torch.eye(4, device=device).unsqueeze(0)
    camera_ids = torch.tensor([0], device=device) if cfg.app_opt else None

    if not cfg.compression_sim or not hasattr(runner, 'comp_sim_splats'):
        selected_anchors = runner.splats["anchors"][visible_anchor_mask]
        selected_offsets = runner.splats["offsets"][visible_anchor_mask]
    else:
        selected_anchors = runner.comp_sim_splats["anchors"][visible_anchor_mask]
        selected_offsets = runner.comp_sim_splats["offsets"][visible_anchor_mask]

    results = runner.decoding_features(
        camtoworlds, time_val, visible_anchor_mask, canonical=False, step=-1, camera_ids=camera_ids
    )

    neural_opacity = results["neural_opacity"]
    primitive_activity_mask = (neural_opacity > 0.0).view(-1)
    
    motion = results["motion"]
    anchor_offset = motion[:, -7:-4]
    moved_anchors = selected_anchors + anchor_offset
    anchor_rot = F.normalize(0.1 * motion[:, -4:] + torch.tensor([[1, 0, 0, 0]], device=device))
    anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)
    
    selected_scales = results["selected_scales"]
    transformed_offsets = torch.bmm(
        selected_offsets.view(-1, cfg.n_offsets, 3) * selected_scales.unsqueeze(1)[:, :, :3],
        anchor_rotation.reshape((-1, 3, 3)).transpose(1, 2),
    ).reshape((-1, 3))

    scales_repeated = selected_scales.unsqueeze(1).repeat(1, cfg.n_offsets, 1).view(-1, 6)
    anchors_repeated = moved_anchors.unsqueeze(1).repeat(1, cfg.n_offsets, 1).view(-1, 3)
    
    neural_scale_rot = results["neural_scale_rot"]
    scales = scales_repeated[:, 3:] * torch.sigmoid(neural_scale_rot[:, :3])
    
    current = {
        "means": anchors_repeated + transformed_offsets,
        "scales": torch.log(scales.clamp(min=1e-8)),
        "quats": F.normalize(neural_scale_rot[:, 3:7]),
        "opacities": torch.logit(neural_opacity.squeeze(-1).clamp(min=0.0, max=1.0)).clamp(min=-12.0, max=12.0),
        "sh0": rgb2sh(results["neural_colors"]).unsqueeze(1),
        "shN": torch.zeros((results["neural_colors"].shape[0], 0, 3), device=device),
    }

    if previous_gaussians is not None:
        inactive_mask = ~primitive_activity_mask
        for key, tensor in current.items():
            tensor[inactive_mask] = previous_gaussians[key][inactive_mask]

    return current, primitive_activity_mask


def load_simple_gop_model(runner: Runner, ckpt_path: str, device: str):
    """Loads a checkpoint from the simple_trainer into the runner."""
    ckpt = torch.load(ckpt_path, map_location=device)
    splats_dict = ckpt.get("splats", {})
    for k in runner.splats.keys():
        if k in splats_dict:
            runner.splats[k].data = splats_dict[k].to(device)
    if "decoders" in ckpt:
        runner.decoders.load_state_dict(ckpt["decoders"])
    runner.decoders.to(device)
    if runner.cfg.knn:
        _, runner.indices = find_k_neighbors(runner.splats["anchors"], runner.cfg.n_knn)

# --- New CLI Configuration and Main Logic ---

@dataclass
class CompressConfig:
    """Configuration for compressing a GIFStream model into video.

    Args:
        checkpoints_dir: Path to the directory containing GOP checkpoints (e.g., ckpt_*.pt).
        config_path: Path to the original model config.yml file.
        output_dir: Path where compressed videos and metadata will be saved.
        sort_keys: Parameters to use for the spatiotemporal sort.
        device: Device to use for processing.
    """
    checkpoints_dir: str = "./checkpoints"
    config_path: str = "./config.yml"
    output_dir: str = "./compressed_output"
    sort_keys: List[str] = field(default_factory=lambda: ["means"])
    device: str = "cuda:0"

@dataclass
class DecompressConfig:
    """Configuration for decompressing videos back into a sequence of PLY files.

    Args:
        input_dir: Path to the directory containing the compressed videos and metadata.
        output_dir: Path where the output PLY files will be saved.
        device: Device to use for processing.
    """
    input_dir: str = "./compressed_output"
    output_dir: str = "./decompressed_plys"
    device: str = "cuda:0"

def run_compression(config: CompressConfig):
    """Orchestrates the compression workflow for GIFStream models."""
    print("--- Starting GIFStream Compression ---")
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Find and sort all checkpoint files
    def gop_sort_key(s):
        match = re.search(r'(\d+)', os.path.basename(s))
        return int(match.group(1)) if match else -1
    gop_checkpoints = sorted(glob.glob(os.path.join(config.checkpoints_dir, "*.pt")), key=gop_sort_key)
    if not gop_checkpoints:
        raise FileNotFoundError(f"No GOP checkpoints (*.pt) found in {config.checkpoints_dir}")
    print(f"Found {len(gop_checkpoints)} GOP checkpoints to process.")

    # 2. Initialize the GIFStream Runner
    with open(config.config_path, 'r') as f:
        trainer_config_dict = yaml.unsafe_load(f)
    cfg = Config()
    for key, value in trainer_config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.disable_viewer = True
    runner = Runner(0, 0, 1, cfg)

    # 3. Process each checkpoint as a chunk
    compressor = Compression(use_sort=True, verbose=False)
    all_processed_chunks_data = []
    previous_sorted_average_frame = None

    for ckpt_path in tqdm(gop_checkpoints, desc="Processing GOPs"):
        load_simple_gop_model(runner, ckpt_path, config.device)
        gop_size = runner.cfg.GOP_size

        # --- CRITICAL: Replicate legacy gap-filling logic BEFORE compression ---
        # This pre-caches all frames for the GOP with gaps filled.
        gop_raw_frames = []
        previous_gaussians = None
        # a. Forward pass to fill from previous frames
        for frame_in_gop in range(gop_size):
            time_val = frame_in_gop / (gop_size - 1) if gop_size > 1 else 0
            gaussians, activity_mask = get_gaussians_for_frame(
                runner, time_val, previous_gaussians=previous_gaussians
            )
            gop_raw_frames.append({"splats": gaussians, "activity_mask": activity_mask})
            previous_gaussians = {k: v.clone() for k, v in gaussians.items()}
        
        # b. Backward pass to fill remaining gaps from next frames
        if gop_size > 1:
            for i in range(gop_size - 2, -1, -1):
                inactive_mask = ~gop_raw_frames[i]["activity_mask"]
                for key in gop_raw_frames[i]["splats"]:
                    current_tensor = gop_raw_frames[i]["splats"][key]
                    next_tensor = gop_raw_frames[i+1]["splats"][key]
                    current_tensor[inactive_mask] = next_tensor[inactive_mask]
        # --- End of gap-filling logic ---

        # c. Define a simple, stateless loader that reads from our cache
        def gop_frame_loader(frame_idx_in_gop: int) -> Dict[str, torch.Tensor]:
            frame_data = gop_raw_frames[frame_idx_in_gop]
            return {**frame_data["splats"], "activity_mask": frame_data["activity_mask"]}

        # d. Use the ChunkProcessor with the clean, pre-processed data
        processor = ChunkProcessor(
            compressor=compressor,
            frame_loader=gop_frame_loader,
            num_frames=gop_size,
            device=config.device,
        )
        
        processed_chunk = processor.run(
            sort_keys=config.sort_keys,
            previous_sorted_grid=None,
            use_average_for_sorting=False,
            use_delta_encoding=False,
            use_morton_sort=True,
        )
        
        all_processed_chunks_data.append(processed_chunk)
        # if processed_chunk["sidelen"] > 0:
        #     previous_sorted_average_frame = processed_chunk["sorted_average_frame"]

    # 4. Finalization: Unify frame sizes and save (copied from legacy)
    print("\n--- Finalizing and Writing Output ---")
    final_video_buffers, full_meta = io_utils.prepare_chunks_for_video_writing(all_processed_chunks_data)

    if not final_video_buffers:
        print("No data to write. Exiting.")
        return

    # Use the library's I/O utility
    io_utils.write_video_streams(config.output_dir, final_video_buffers, codec="libx265")
    io_utils.write_metadata(os.path.join(config.output_dir, "meta_bundle.json"), full_meta)
    print("\n--- Compression Complete ---")


def run_decompression(config: DecompressConfig):
    """Orchestrates the decompression workflow."""
    print("--- Starting Decompression ---")
    os.makedirs(config.output_dir, exist_ok=True)

    player = DecompressionPlayer(config.input_dir, device=config.device)

    for i in tqdm(range(len(player)), desc="Decompressing Frames"):
        splats = player.get_splats_for_frame(i)
        
        if splats and splats.get("means") is not None and splats["means"].shape[0] > 0:
            output_ply_path = os.path.join(config.output_dir, f"frame_{i:05d}.ply")
            export_splats(save_to=output_ply_path, **splats)

    print("\n--- Decompression Complete ---")


def cli_main():
    """Main entry point that dispatches to the correct mode."""
    configs = {
        "compress": (
            "Compress a GIFStream model into video format.",
            CompressConfig(),
        ),
        "decompress": (
            "Decompress video format back into a sequence of PLY files.",
            DecompressConfig(),
        ),
    }

    config_choice = tyro.extras.overridable_config_cli(configs)

    if isinstance(config_choice, CompressConfig):
        run_compression(config_choice)
    elif isinstance(config_choice, DecompressConfig):
        run_decompression(config_choice)

if __name__ == "__main__":
    cli_main()