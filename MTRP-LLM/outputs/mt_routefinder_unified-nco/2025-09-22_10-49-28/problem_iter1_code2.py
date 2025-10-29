import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute a modified version of the normalized distance-based heuristic score matrix for improved edge selection
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Compute a modified version of the demand-based heuristic score matrix for diverse exploration
    demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the modified heuristic scores for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remaining parts of the function remain unchanged from the original heuristics_v1 function

    return cvrp_scores