import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Calculate inverse of distance matrix with epsilon smoothing
    distance_inverse = 1 / (current_distance_matrix + 1e-8)

    # Calculate load ratio for each trajectory with epsilon smoothing
    load_ratio = current_load.unsqueeze(-1) / (delivery_node_demands + 1e-8)

    # Calculate open route load ratio for each trajectory with epsilon smoothing
    load_open_ratio = current_load_open.unsqueeze(-1) / (delivery_node_demands_open + 1e-8)

    # Random noise for controlled randomness
    random_noise = torch.rand_like(current_distance_matrix)

    # Heuristic score calculation combining different metrics
    heuristic_score = distance_inverse * load_ratio + load_open_ratio - random_noise

    return heuristic_score