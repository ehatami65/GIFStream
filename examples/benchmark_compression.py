import json
import os
import time
import shutil
from dataclasses import dataclass
from typing import Dict
import math

import torch
import tyro
import yaml
import psutil
import numpy as np
import tqdm

from simple_trainer_GIFStream import Runner, Config
from gsplat.compression import GIFStreamEnd2endCompression, GIFStream2dcodecCompression
from utils import find_k_neighbors


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking script."""
    # Path to the config.yml file.
    config_path: str = "/data/shared/aly/results/cut_roastbeef_50k/cfg.yml"
    # Path to the checkpoint to benchmark.
    ckpt: str = "/data/shared/aly/results/cut_roastbeef_50k/ckpts/ckpt_49999_rank0.pt"
    # Directory to save benchmark results.
    benchmark_dir: str = "/data/shared/aly/results/cut_roastbeef_50k/benchmark_compression_GPU"


@torch.no_grad()
class Benchmarker:
    def __init__(self, cfg: BenchmarkConfig):
        self.benchmark_cfg = cfg
        self.device = "cuda:0"
        os.makedirs(cfg.benchmark_dir, exist_ok=True)
        print(f"Benchmark results will be saved to {cfg.benchmark_dir}")

        # Load trainer config from yml
        with open(cfg.config_path, 'r') as f:
            # Use unsafe_load to handle python objects like GIFStreamStrategy
            trainer_config_dict = yaml.unsafe_load(f)

        # Create a Config object and update it with the loaded dict
        self.trainer_cfg = Config()
        for key, value in trainer_config_dict.items():
            if hasattr(self.trainer_cfg, key):
                setattr(self.trainer_cfg, key, value)

        # We need to suppress viewer from Runner initialization
        self.trainer_cfg.disable_viewer = True
        self.trainer_cfg.result_dir = self.benchmark_cfg.benchmark_dir
        
        # We will initialize the runner when needed to ensure clean state
        self.runner = None

        # Initialize compression methods
        self.compression_method_end2end = GIFStreamEnd2endCompression()
        self.compression_method_2dcodec = GIFStream2dcodecCompression()

    def setup_runner(self):
        """Initializes a fresh runner and loads the checkpoint."""
        print("\n--- Setting up a fresh runner and loading checkpoint ---")
        self.runner = Runner(0, 0, 1, self.trainer_cfg)
        self.runner.device = self.device
        
        self.runner.decoders.to(self.device)
        if hasattr(self.runner, "entropy_models"):
            for model in self.runner.entropy_models.values():
                if model is not None:
                    model.to(self.device)
        for param in self.runner.splats.values():
            param.data = param.data.to(self.device)

        if hasattr(self.runner, "app_module") and self.runner.app_module is not None:
            self.runner.app_module.to(self.device)

        self.load_checkpoint()

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.benchmark_cfg.ckpt}")
        ckpt = torch.load(self.benchmark_cfg.ckpt, map_location=self.device)
        
        splats_dict = ckpt.get("splats", {})
        for k in self.runner.splats.keys():
            if k in splats_dict:
                self.runner.splats[k].data = splats_dict[k]
        
        if "decoders" in ckpt:
            self.runner.decoders.load_state_dict(ckpt["decoders"])

        if "app_module" in ckpt and self.runner.app_module is not None:
            self.runner.app_module.load_state_dict(ckpt["app_module"])
        
        self.runner.cfg.compression_sim = ckpt.get("compression_sim", self.runner.cfg.compression_sim)
        if self.runner.cfg.compression_sim:
            self.runner.load_entropy_model_from_ckpt(ckpt, self.trainer_cfg.entropy_model_type)
        
        if self.trainer_cfg.knn:
            _, self.runner.indices = find_k_neighbors(self.runner.splats["anchors"], self.trainer_cfg.n_knn)
        print("Checkpoint loaded.")

    def run(self):
        """Main entry point for benchmarking."""

        # 0. Create ply files for baseline
        self.setup_runner()
        print("\n--- Exporting baseline PLY sequence (step 0) ---")
        self.runner.export_ply_sequence(0)

        # --- Compression Phase ---
        end2end_compression_report = self.benchmark_compression_end2end()
        
        self.setup_runner() # Reset runner state
        codec2d_compression_report = self.benchmark_compression_2dcodec()

        # --- Decompression and Evaluation for End-to-End ---
        print("\n\n--- Decompressing and Evaluating End-to-End ---")
        self.setup_runner() # Fresh runner for clean state
        end2end_decompression_report = self.benchmark_decompression_end2end()
        print("\n--- Exporting end2end decompressed PLY sequence (step 1) ---")
        self.runner.export_ply_sequence(1)
        end2end_splats_to_gaussians_report = self.benchmark_splats_to_gaussians()

        # --- Decompression and Evaluation for 2D Codec ---
        print("\n\n--- Decompressing and Evaluating 2D Codec ---")
        self.setup_runner() # Another fresh runner
        codec2d_decompression_report = self.benchmark_decompression_2dcodec()
        print("\n--- Exporting 2dcodec decompressed PLY sequence (step 2) ---")
        self.runner.export_ply_sequence(2)

        # Final report
        final_report = {
            "end2end_compression": end2end_compression_report,
            "2dcodec_compression": codec2d_compression_report,
            "end2end_decompression": end2end_decompression_report,
            "2dcodec_decompression": codec2d_decompression_report,
            "splats_to_gaussians_conversion (from end2end)": end2end_splats_to_gaussians_report,
        }

        report_path = os.path.join(self.benchmark_cfg.benchmark_dir, "benchmark_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
        
        print(f"\nBenchmark finished. Report saved to {report_path}")

    def benchmark_compression_end2end(self):
        print("\n--- Benchmarking End-to-End Compression ---")
        compress_dir = os.path.join(self.benchmark_cfg.benchmark_dir, "compressed_data_end2end")
        if os.path.exists(compress_dir): shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        report = {"time": {}, "memory": {}, "cpu": {}}
        p = psutil.Process(os.getpid())
        
        torch.cuda.synchronize()
        start_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            start_cpu_times = p.cpu_times()
            start_ram_mem = p.memory_info().rss
        start_time = time.time()

        splats_to_compress = {k: v.clone() for k, v in self.runner.comp_sim_splats.items()}
        self.compression_method_end2end.compress(
            compress_dir, splats_to_compress, self.runner.entropy_models,
            self.trainer_cfg.entropy_channel, self.trainer_cfg.c_perframe,
            self.runner.scaling, self.trainer_cfg.voxel_size
        )

        nets = {
            "decoders": self.runner.decoders.state_dict(),
            "scaling": self.runner.scaling
        }
        for name, entropy_model in self.runner.entropy_models.items():
            if entropy_model is not None:
                nets[name + "_entropy_model"] = entropy_model.state_dict()
        torch.save(nets, os.path.join(compress_dir, "nets.pt"))

        torch.cuda.synchronize()
        end_time = time.time()
        
        end_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            end_cpu_times = p.cpu_times()
            end_ram_mem = p.memory_info().rss
        
        report["time"]["total_compression_time_seconds"] = end_time - start_time
        report["memory"]["gpu_mem_used_bytes"] = end_gpu_mem - start_gpu_mem
        report["memory"]["ram_mem_used_bytes"] = end_ram_mem - start_ram_mem
        report["cpu"]["cpu_time_seconds"] = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)

        print(f"Total compression time: {report['time']['total_compression_time_seconds']:.4f}s")
        return report
    
    def benchmark_compression_2dcodec(self):
        print("\n--- Benchmarking 2D Codec Compression ---")
        compress_dir = os.path.join(self.benchmark_cfg.benchmark_dir, "compressed_data_2dcodec")
        if os.path.exists(compress_dir): shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        report = {"time": {}, "memory": {}, "cpu": {}}
        p = psutil.Process(os.getpid())
        
        torch.cuda.synchronize()
        start_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            start_cpu_times = p.cpu_times()
            start_ram_mem = p.memory_info().rss
        start_time = time.time()
        
        splats_to_compress = {k: v.clone() for k, v in self.runner.comp_sim_splats.items()}
        self.compression_method_2dcodec.compress(compress_dir, splats_to_compress)

        torch.cuda.synchronize()
        end_time = time.time()
        
        end_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            end_cpu_times = p.cpu_times()
            end_ram_mem = p.memory_info().rss
        
        report["time"]["total_compression_time_seconds"] = end_time - start_time
        report["memory"]["gpu_mem_used_bytes"] = end_gpu_mem - start_gpu_mem
        report["memory"]["ram_mem_used_bytes"] = end_ram_mem - start_ram_mem
        report["cpu"]["cpu_time_seconds"] = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)

        print(f"Total compression time: {report['time']['total_compression_time_seconds']:.4f}s")
        return report

    def benchmark_decompression_end2end(self):
        print("\n--- Benchmarking End-to-End Decompression ---")
        compress_dir = os.path.join(self.benchmark_cfg.benchmark_dir, "compressed_data_end2end")
        report = {"time": {}, "memory": {}, "cpu": {}}
        p = psutil.Process(os.getpid())

        torch.cuda.synchronize()
        start_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            start_cpu_times = p.cpu_times()
            start_ram_mem = p.memory_info().rss

        start_time = time.time()
        
        self.runner.load_models_from_compressed_dir(compress_dir, self.trainer_cfg.entropy_model_type)
        splats_c = self.compression_method_end2end.decompress(
            compress_dir, self.runner.entropy_models, self.device
        )
        for k, v in splats_c.items():
            if k in self.runner.splats:
                self.runner.splats[k].data = v.to(self.device)

        self.runner.cfg.compression_sim = False
        
        torch.cuda.synchronize()
        end_time = time.time()

        end_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            end_cpu_times = p.cpu_times()
            end_ram_mem = p.memory_info().rss

        report["time"]["total_decompression_time_seconds"] = end_time - start_time
        report["memory"]["gpu_mem_used_bytes"] = end_gpu_mem - start_gpu_mem
        report["memory"]["ram_mem_used_bytes"] = end_ram_mem - start_ram_mem
        report["cpu"]["cpu_time_seconds"] = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
        
        print(f"Total decompression time: {report['time']['total_decompression_time_seconds']:.4f}s")
        return report

    def benchmark_decompression_2dcodec(self):
        print("\n--- Benchmarking 2D Codec Decompression ---")
        compress_dir = os.path.join(self.benchmark_cfg.benchmark_dir, "compressed_data_2dcodec")
        report = {"time": {}, "memory": {}, "cpu": {}}
        p = psutil.Process(os.getpid())

        torch.cuda.synchronize()
        start_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            start_cpu_times = p.cpu_times()
            start_ram_mem = p.memory_info().rss

        start_time = time.time()
        
        splats_c = self.compression_method_2dcodec.decompress(compress_dir)
        for k, v in splats_c.items():
            if k in self.runner.splats:
                self.runner.splats[k].data = v.to(self.device)

        self.runner.cfg.compression_sim = False
        
        torch.cuda.synchronize()
        end_time = time.time()

        end_gpu_mem = torch.cuda.memory_allocated()
        with p.oneshot():
            end_cpu_times = p.cpu_times()
            end_ram_mem = p.memory_info().rss

        report["time"]["total_decompression_time_seconds"] = end_time - start_time
        report["memory"]["gpu_mem_used_bytes"] = end_gpu_mem - start_gpu_mem
        report["memory"]["ram_mem_used_bytes"] = end_ram_mem - start_ram_mem
        report["cpu"]["cpu_time_seconds"] = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
        
        print(f"Total decompression time: {report['time']['total_decompression_time_seconds']:.4f}s")
        return report

    def benchmark_splats_to_gaussians(self):
        print("\n--- Benchmarking Splat-to-Gaussian Conversion (averaged over all time steps) ---")
        report = {"time": {}, "memory": {}, "cpu": {}}
        p = psutil.Process(os.getpid())
        
        camtoworlds = torch.eye(4, device=self.device).unsqueeze(0)
        camera_ids = torch.tensor([0], device=self.device) if self.runner.cfg.app_opt else None

        gop_size = self.runner.cfg.GOP_size
        
        total_times = []
        total_ram_mem_diffs = []
        total_cpu_time_diffs = []
        
        for frame_idx in tqdm.trange(gop_size, desc="Benchmarking splat-to-gaussian conversion"):
            time_val = frame_idx / (gop_size - 1)
            
            with p.oneshot():
                start_cpu_times = p.cpu_times()
                start_ram_mem = p.memory_info().rss
            start_time = time.time()
            
            # This logic is adapted from export_ply_sequence to avoid CUDA calls
            visible_anchor_mask = torch.ones(self.runner.splats["anchors"].shape[0], dtype=torch.bool, device=self.device)
            
            selected_anchors = self.runner.splats["anchors"][visible_anchor_mask]
            selected_offsets = self.runner.splats["offsets"][visible_anchor_mask]

            results = self.runner.decoding_features(
                camtoworlds, time_val, visible_anchor_mask, canonical=False, step=-1, camera_ids=camera_ids
            )
            
            motion = results["motion"]
            selected_scales = results["selected_scales"]
            
            anchor_offset = motion[:, -7:-4]
            moved_anchors = selected_anchors + anchor_offset
            anchor_rot = torch.nn.functional.normalize(
                0.1 * motion[:, -4:] + torch.tensor([[1, 0, 0, 0]], device=self.device)
            )
            anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)
            
            transformed_offsets = torch.bmm(
                selected_offsets.view(-1, self.runner.cfg.n_offsets, 3) * selected_scales.unsqueeze(1)[:, :, :3],
                anchor_rotation.reshape((-1, 3, 3)).transpose(1, 2),
            ).reshape((-1, 3))
            
            anchors_repeated = (
                moved_anchors.unsqueeze(1).repeat(1, self.runner.cfg.n_offsets, 1).view(-1, 3)
            )
            
            _ = anchors_repeated + transformed_offsets # means
            
            end_time = time.time()
            with p.oneshot():
                end_cpu_times = p.cpu_times()
                end_ram_mem = p.memory_info().rss

            total_times.append(end_time - start_time)
            total_ram_mem_diffs.append(end_ram_mem - start_ram_mem)
            cpu_time_seconds = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
            total_cpu_time_diffs.append(cpu_time_seconds)

        report["time"]["avg_conversion_time_seconds"] = np.mean(total_times)
        report["memory"]["avg_ram_mem_used_bytes"] = np.mean(total_ram_mem_diffs)
        report["cpu"]["avg_cpu_time_seconds"] = np.mean(total_cpu_time_diffs)

        print(f"Average splat-to-gaussian conversion time: {report['time']['avg_conversion_time_seconds']:.4f}s")
        print(f"Average RAM used: {report['memory']['avg_ram_mem_used_bytes'] / 1e9:.4f} GB")
        print(f"Average CPU time: {report['cpu']['avg_cpu_time_seconds']:.4f}s")

        return report

def quaternion_to_rotation_matrix(quaternion):
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    
    w, x, y, z = quaternion.unbind(dim=-1)
    
    B = quaternion.size(0)
    
    rotation_matrix = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
    ], dim=-1).view(B, 3, 3)
    
    return rotation_matrix

def benchmark_main(cfg: BenchmarkConfig):
    benchmarker = Benchmarker(cfg)
    benchmarker.run()

if __name__ == "__main__":
    cfg = tyro.cli(BenchmarkConfig)
    benchmark_main(cfg) 