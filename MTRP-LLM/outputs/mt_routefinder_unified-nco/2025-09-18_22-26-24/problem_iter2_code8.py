import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate heuristic indicators based on node properties
    heuristic_indicators = torch.rand_like(current_distance_matrix)

    # Introduce randomness by shuffling the heuristic indicators
    shuffled_indicators = heuristic_indicators.clone()
    permutation = torch.randperm(shuffled_indicators.shape[1])
    shuffled_indicators = shuffled_indicators[:, permutation]

    return shuffled_indicators