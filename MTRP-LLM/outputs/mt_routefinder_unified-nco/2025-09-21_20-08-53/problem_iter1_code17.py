import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the squared and inverted distance-based heuristic score matrix with added diversity through randomness
    squared_distance_scores = -torch.pow(current_distance_matrix, 2) / torch.max(torch.pow(current_distance_matrix, 2)) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Compute the inverse demand-based heuristic score matrix with emphasis on high-demand nodes and enhanced randomness
    inverse_demand_scores = 1 / (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1) + 1e-6) + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = squared_distance_scores + inverse_demand_scores + enhanced_noise

    # Remainder of the code remains unchanged

    return cvrp_scores