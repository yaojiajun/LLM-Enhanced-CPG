import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Incorporate problem-specific information for more targeted exploration and exploitation
    exploration_factor = torch.rand_like(current_distance_matrix) * 0.05
    exploitation_factor = torch.randn_like(current_distance_matrix) * 0.1

    heuristic_scores = current_distance_matrix * (1 + exploration_factor) + exploitation_factor

    return heuristic_scores