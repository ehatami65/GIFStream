from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from typing_extensions import Literal
from torch_scatter import scatter_max
from .base import Strategy
from .ops import _update_param_with_optimizer

@dataclass
class StatefulGIFStreamStrategy(Strategy):
    """
    A stateful strategy for GIFStream that handles a fixed-size pool of anchors,
    enabling continuous training over long sequences. It activates and deactivates
    anchors from the pool instead of adding or removing them, preserving tensor
    sizes and order for video compression.
    """
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4
    densify_grad_threshold: float = 0.00009
    prune_opa: float = 0.005
    success_threshold: float = 0.8
    check_interval: int = 100
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100

    absgrad: bool = False
    verbose: bool = True
    key_for_gradient: Literal["means2d"] = "means2d"

    def initialize_state(self, scene_scale: float, n_offsets: int, voxel_size: float,
                         active_mask: torch.Tensor, static_slots: torch.Tensor,
                         dynamic_slots: torch.Tensor, gop_size: int, max_total_anchors: int) -> Dict[str, Any]:
        """Initializes the strategy's state tensors to the full anchor pool size."""
        self.active_mask = active_mask
        self.static_slots = static_slots
        self.dynamic_slots = dynamic_slots
        self.n_offsets = n_offsets
        self.max_total_anchors = max_total_anchors
        
        device = active_mask.device
        state = {
            "scene_scale": scene_scale,
            "voxel_size": voxel_size,
            "offset_grad2d": torch.zeros((max_total_anchors * self.n_offsets, gop_size), device=device),
            "offset_demon": torch.zeros((max_total_anchors * self.n_offsets, gop_size), device=device),
            "opacity_accum": torch.zeros((max_total_anchors, gop_size), device=device),
            "anchor_demon": torch.zeros((max_total_anchors, gop_size), device=device),
        }
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        required_keys = ["anchors", "scales", "opacities", "offsets", "anchor_features", "time_features", "factors"]
        for key in required_keys:
            assert key in params, f"'{key}' is required in params but missing."

    def step_pre_backward(self, params, optimizers, state, step, info):
        # CORRECTED: The 'info' object is a single dictionary for one item in the batch.
        # The for loop was incorrect and caused the TypeError.
        if info is not None and self.key_for_gradient in info:
            info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, gop_index):
        if step < self.refine_start_iter or step >= self.refine_stop_iter:
            return

        self._update_state(state, info)

        if step % self.refine_every == 0:
            self._prune_and_grow_gs(params, optimizers, state, gop_index)
            torch.cuda.empty_cache()

    def _update_state(self, state: Dict[str, Any], batch_info: List[Dict[str, Any]]):
        for info in batch_info:
            if info is None: continue
            if info[self.key_for_gradient].grad is None: continue
            
            batch_size = len(batch_info)
            grads = info[self.key_for_gradient].grad.clone().squeeze(0)
            grads[..., 0] *= info["width"] / 2.0
            grads[..., 1] *= info["height"] / 2.0
            grad_norm = grads.norm(dim=-1)
            
            update_filter = info["update_filter"].squeeze(0)
            global_offset_indices = info["global_offset_indices"]
            
            rendered_offset_indices = global_offset_indices[update_filter]
            rendered_grad_norm = grad_norm[update_filter]
            
            state["offset_demon"][:, info["time"]].index_add_(
                0, rendered_offset_indices, torch.ones_like(rendered_offset_indices, dtype=torch.float32) / batch_size
            )
            state["offset_grad2d"][:, info["time"]].index_add_(
                0, rendered_offset_indices, rendered_grad_norm / batch_size
            )
            
            visible_anchor_global_indices = info["anchor_visible_mask"].nonzero(as_tuple=False).squeeze(-1)

            if visible_anchor_global_indices.numel() > 0:
                temp_opacity = info["neural_opacity"].clone().view(-1).detach()
                temp_opacity[temp_opacity < 0] = 0
                temp_opacity = temp_opacity.view(-1, self.n_offsets)
                opacity_sum_for_visible_anchors = temp_opacity.sum(dim=1)

                state["opacity_accum"][:, info["time"]].index_add_(
                    0, visible_anchor_global_indices, opacity_sum_for_visible_anchors / batch_size
                )
                state["anchor_demon"][:, info["time"]].index_add_(
                    0, visible_anchor_global_indices, torch.ones_like(visible_anchor_global_indices, dtype=torch.float32) / batch_size
                )

    @torch.no_grad()
    def _prune_and_grow_gs(self, params, optimizers, state, gop_index):
        # --- 1. PRUNING ---
        active_indices = torch.where(self.active_mask)[0]
        if active_indices.numel() == 0: return

        anchor_demon_active = state["anchor_demon"][active_indices]
        prune_seen_mask = anchor_demon_active.sum(dim=-1) > self.check_interval * self.success_threshold
        
        evaluated_prune_indices = active_indices[prune_seen_mask]
        num_pruned = 0

        if evaluated_prune_indices.numel() > 0:
            opacity_accum_seen = state["opacity_accum"][evaluated_prune_indices]
            avg_opacity = opacity_accum_seen.sum(dim=-1) / anchor_demon_active[prune_seen_mask].sum(dim=-1).clamp(min=1)
            low_opacity_mask = avg_opacity < self.prune_opa
            
            global_prune_indices = evaluated_prune_indices[low_opacity_mask]
            num_pruned = global_prune_indices.numel()

            if num_pruned > 0:
                self.active_mask[global_prune_indices] = False
                self.static_slots[global_prune_indices] = False
                self.dynamic_slots[global_prune_indices] = False

        if evaluated_prune_indices.numel() > 0:
            state["opacity_accum"][evaluated_prune_indices].zero_()
            state["anchor_demon"][evaluated_prune_indices].zero_()
        
        # --- 2. DENSIFICATION ---
        total_densified = 0
        if self.active_mask.sum() < self.max_total_anchors:
            eligible_for_densification_mask = self.dynamic_slots if gop_index > 0 else self.active_mask
            active_and_eligible_indices = torch.where(self.active_mask & eligible_for_densification_mask)[0]

            if active_and_eligible_indices.numel() > 0:
                offset_starts = active_and_eligible_indices * self.n_offsets
                offset_ranges = torch.arange(self.n_offsets, device=active_and_eligible_indices.device)
                eligible_offset_indices = (offset_starts.unsqueeze(1) + offset_ranges).view(-1)
                
                offset_grad_eligible = state["offset_grad2d"][eligible_offset_indices]
                offset_demon_eligible = state["offset_demon"][eligible_offset_indices]
                offset_seen_mask_flat = offset_demon_eligible.sum(-1) > self.check_interval * self.success_threshold * 0.5
                
                grads_flat = torch.zeros_like(offset_grad_eligible.sum(-1))
                if offset_seen_mask_flat.any():
                    peak_ratio = 0.1
                    seen_grads = offset_grad_eligible[offset_seen_mask_flat]
                    seen_demons = offset_demon_eligible[offset_seen_mask_flat]
                    weighted_grads = peak_ratio * (seen_grads / seen_demons.clamp(min=1e-7)) + \
                                     (1 - peak_ratio) * seen_grads.sum(-1, keepdim=True) / seen_demons.sum(-1, keepdim=True).clamp(min=1e-7)
                    grads_flat[offset_seen_mask_flat] = weighted_grads.nan_to_num(0.0).max(-1)[0]
                
                for i in range(self.update_depth):
                    if self.active_mask.sum() >= self.max_total_anchors: break
                    cur_threshold = self.densify_grad_threshold * ((self.update_hierachy_factor // 2) ** i)
                    candidate_mask_flat = (grads_flat >= cur_threshold) & offset_seen_mask_flat
                    if not candidate_mask_flat.any(): continue
                    
                    all_xyz = (params["anchors"][active_and_eligible_indices].unsqueeze(1) + 
                               params["offsets"][active_and_eligible_indices] * torch.exp(params["scales"][active_and_eligible_indices, :3]).unsqueeze(1)).view(-1, 3)
                    selected_xyz = all_xyz[candidate_mask_flat]
                    
                    size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
                    cur_size = state["voxel_size"] * size_factor
                    
                    selected_grid_coords = torch.round(selected_xyz / cur_size).int()
                    selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
                    
                    active_anchor_grid_coords = torch.round(params["anchors"][self.active_mask] / cur_size).int()
                    is_occupied = (selected_grid_coords_unique.unsqueeze(1) == active_anchor_grid_coords).all(-1).any(-1)
                    new_anchor_positions = selected_grid_coords_unique[~is_occupied] * cur_size

                    num_new = new_anchor_positions.shape[0]
                    if num_new == 0: continue

                    free_indices = torch.where(~self.active_mask)[0]
                    num_to_add = min(num_new, free_indices.numel())
                    if num_to_add == 0: break
                    new_indices = free_indices[:num_to_add]
                    
                    primitive_indices_that_spawned = candidate_mask_flat.nonzero(as_tuple=False).squeeze(-1)
                    parent_indices_of_primitives = active_and_eligible_indices.repeat_interleave(self.n_offsets)[primitive_indices_that_spawned]
                    
                    parent_grads = grads_flat[primitive_indices_that_spawned]
                    _, best_primitive_for_cell_idx = scatter_max(parent_grads, inverse_indices, dim=0)
                    best_primitive_for_cell_idx = best_primitive_for_cell_idx[~is_occupied][:num_to_add]
                    
                    parent_indices_to_clone = parent_indices_of_primitives[best_primitive_for_cell_idx]
                    
                    new_params = {
                        "anchors": new_anchor_positions[:num_to_add],
                        "scales": params["scales"][parent_indices_to_clone].clone(),
                        "quats": params["quats"][parent_indices_to_clone].clone(),
                        # CORRECTED TYPO: num_to__add -> num_to_add
                        "opacities": torch.logit(0.1 * torch.ones(num_to_add, 1, device=params["anchors"].device)),
                        "offsets": torch.zeros(num_to_add, self.n_offsets, 3, device=params["anchors"].device),
                        "anchor_features": params["anchor_features"][parent_indices_to_clone].clone(),
                        "factors": params["factors"][parent_indices_to_clone].clone(),
                        "time_features": torch.zeros(num_to_add, params["time_features"].shape[1], params["time_features"].shape[2], device=params["anchors"].device)
                    }
                    self.add_anchors(params, optimizers, state, new_params, new_indices)
                    total_densified += num_to_add
                    grads_flat[candidate_mask_flat] = 0.0

                evaluated_offset_indices = eligible_offset_indices[offset_seen_mask_flat]
                if evaluated_offset_indices.numel() > 0:
                    state["offset_grad2d"][evaluated_offset_indices] = 0.0
                    state["offset_demon"][evaluated_offset_indices] = 0.0

        if self.verbose:
            print(f"Pruned {num_pruned} anchors. Densified {total_densified} new anchors. Total active: {self.active_mask.sum().item()}")


    def add_anchors(self, params, optimizers, state, new_params, indices):
        num_new = len(indices)
        if num_new == 0: return

        for name, data in new_params.items():
            params[name].data[indices] = data

        for name, optimizer in optimizers.items():
            if name in new_params:
                param_state = optimizer.state[params[name]]
                for key in param_state:
                    if key != 'step' and isinstance(param_state[key], torch.Tensor):
                        param_state[key][indices] = 0.0
        
        state["opacity_accum"][indices].zero_()
        state["anchor_demon"][indices].zero_()
        offset_starts = indices * self.n_offsets
        offset_ranges = torch.arange(self.n_offsets, device=indices.device)
        global_offset_indices_to_reset = (offset_starts.unsqueeze(1) + offset_ranges).view(-1)
        state["offset_grad2d"][global_offset_indices_to_reset].zero_()
        state["offset_demon"][global_offset_indices_to_reset].zero_()
        
        self.active_mask[indices] = True
        self.static_slots[indices] = False
        self.dynamic_slots[indices] = True