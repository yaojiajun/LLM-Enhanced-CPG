import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Compute heuristic scores based on penalty-based approaches and enhanced randomness
    penalty_factor = 0.5
    heuristic_scores = torch.rand_like(current_distance_matrix) * 2 - 1  # Random scores between -1 and 1
    heuristic_scores -= penalty_factor  # Apply penalty factor

    return heuristic_scores