import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Enhance randomness
    randomness_factor = torch.rand(current_distance_matrix.size())

    # Compute heuristic scores using random factors
    heuristic_scores = current_distance_matrix + randomness_factor

    return heuristic_scores