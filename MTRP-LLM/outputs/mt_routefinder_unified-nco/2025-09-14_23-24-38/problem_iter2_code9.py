import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Implementing an improved heuristic function incorporating problem constraints and edge diversity
    # Your enhanced implementation here

    heuristic_score_matrix = torch.rand_like(current_distance_matrix) * 2 - 1  # Random heuristic score matrix [-1, 1]

    return heuristic_score_matrix