import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score matrix computation for edge selection
    normalized_distance_scores = current_distance_matrix / torch.max(current_distance_matrix) - torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score matrix for promising edge evaluations
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    # Retain enhanced noise level for diversity and exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the updated heuristic scores for improved edge selection
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Include original computations for other heuristic score computations

    return cvrp_scores