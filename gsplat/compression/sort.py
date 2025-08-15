from typing import Dict, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from .temporal_plas import sort_with_static_target


def sort_splats(
    splats: Dict[str, Tensor],
    prev_sorted_grid: Optional[Tensor] = None,
    verbose: bool = True,
    seed: int = None,
    initial_indices: Optional[Tensor] = None,
    improvement_break: float = 1e-4,
) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    Sorts splats using a two-stage hybrid approach for temporal coherence.
    """
    try:
        from plas import sort_with_plas
    except ImportError:
        # ... (error handling)
        pass
    if seed is not None:
        torch.manual_seed(seed)

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    sort_keys = ["means"]
    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)

    # --- Stage 1: Determine the initial layout (the shuffle) ---
    # This logic is now combined with the new `sort_splats` you provided.
    
    if initial_indices is not None:
        shuffled_indices = initial_indices
    else:
        # If no explicit indices, start with a random permutation
        shuffled_indices = torch.randperm(
            params_to_sort.shape[0], device=params_to_sort.device
        )

    # Apply the initial shuffle
    params_to_sort_shuffled = params_to_sort[shuffled_indices]
    
    # Reshape into the correct (C, H, W) format for ALL PLAS functions
    grid = params_to_sort_shuffled.reshape((n_sidelen, n_sidelen, -1)).permute(2, 0, 1)

    # --- Stage 2: Refinement ---
    if prev_sorted_grid is not None:
        print("Using temporally-anchored sorting against static target.")
        
        # Resize target to match current grid size
        target_grid = F.interpolate(
            prev_sorted_grid.unsqueeze(0),
            size=(n_sidelen, n_sidelen),
            mode='nearest' # Use nearest neighbor to preserve features
        ).squeeze(0)

        # Call our custom temporal sorter
        _, sorted_indices_grid = sort_with_static_target(
            grid, # Already in (C, H, W)
            target_grid,
            improvement_break=improvement_break,
            verbose=verbose,
            seed=seed,
        )
        sorted_indices_relative = sorted_indices_grid.squeeze().flatten()
    else:
        # Fallback to standard PLAS
        _, sorted_indices_relative = sort_with_plas(
            grid, # Already in (C, H, W)
            improvement_break=improvement_break, verbose=verbose, seed=seed
        )
        sorted_indices_relative = sorted_indices_relative.squeeze().flatten()
        
    final_indices = shuffled_indices[sorted_indices_relative]

    for k, v in splats.items():
        splats[k] = v[final_indices]
        
    return splats, final_indices


def sort_anchors(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats with Parallel Linear Assignment Sorting from the paper `Compact 3D Scene Representation via
    Self-Organizing Gaussian Grids <https://arxiv.org/pdf/2312.13299>`_.

    .. warning::
        PLAS must installed to use sorting.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    try:
        from plas import sort_with_plas
    except:
        raise ImportError(
            "Please install PLAS with 'pip install git+https://github.com/fraunhoferhhi/PLAS.git' to use sorting"
        )

    n_gs = len(splats["anchors"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    sort_keys = [k for k in splats if k != "time_features"]
    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten().to(torch.long)
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats

