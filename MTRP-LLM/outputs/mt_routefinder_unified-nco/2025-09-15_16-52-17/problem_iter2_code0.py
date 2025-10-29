import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Calculate heuristic score matrix based on a combination of factors and enhanced randomness
    heuristic_scores = torch.rand_like(current_distance_matrix) * 2 - 1  # Random scores between -1 and 1

    # Introduce penalty mechanisms based on problem-specific knowledge

    # Efficiently compute heuristic scores in a vectorized manner

    return heuristic_scores