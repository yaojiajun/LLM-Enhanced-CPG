import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modified version: Compute the distance-based heuristic score matrix with added diversity through randomness and emphasis on load capacity
    normalized_distance_scores = -(current_distance_matrix * 0.9) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Calculate load capacity sensitivity for high-demand nodes
    load_capacity_sensitivity = (100 - current_load)  # Assume vehicle capacity is 100

    # Combining distance and load sensitivity with enhanced randomness
    cvrp_scores = normalized_distance_scores + load_capacity_sensitivity.unsqueeze(1) + torch.randn_like(current_distance_matrix) * 0.5

    # Keeping the rest of the implementation intact

    return cvrp_scores