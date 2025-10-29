import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristic computation for distance scores with squared normalization and added exploration
    normalized_distance_scores = -(current_distance_matrix ** 2) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.9

    # Modified demand scores with absolute difference and scaled standard deviations
    demand_scores = torch.abs(delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.std(
        delivery_node_demands) / 3.0 * torch.randn_like(current_distance_matrix) * 0.4

    # Unchanged heuristic scores
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combined heuristic scores
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remaining part of the original function for other heuristic score computations

    return cvrp_scores