import torch
import numpy as np
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # cvrp_scores
    # Distance-based heuristic with adjusted divergence through randomness
    normalized_distance_scores = current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.3

    # Modified demand constraints scoring emphasizing high-demand nodes with square function
    demand_scores = (delivery_node_demands.unsqueeze(0)**2 - current_load.unsqueeze(1)**2) * 0.7 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Increased noise for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.2

    # Integration of different scores for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep the logic related to other inputs unchanged

    # Return overall scores
    return cvrp_scores