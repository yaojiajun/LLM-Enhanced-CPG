import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modified computation of distance-based heuristic score with a different relationship
    normalized_distance_scores = -current_distance_matrix / (torch.std(current_distance_matrix) + 1e-6) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Modified demand-based heuristic score calculation with a different weighting and randomness
    demand_scores = (2 * delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with a different noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remaining parts of the function remain unchanged
    ...

    return overall_scores