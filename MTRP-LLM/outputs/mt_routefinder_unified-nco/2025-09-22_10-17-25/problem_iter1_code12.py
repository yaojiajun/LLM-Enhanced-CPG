import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores (Modified)
    # Compute the normalized demand-based heuristic score matrix with added randomness
    normalized_demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Compute the distance-based heuristic score matrix with increased diversity through randomness
    distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.9

    # Introduce additional noise for exploration with varied noise levels
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine different heuristic scores with diverse strategies for balanced exploration
    cvrp_scores = normalized_demand_scores + distance_scores + enhanced_noise

    # Leave other parts unchanged from the original heuristics_v1 implementation

    return cvrp_scores