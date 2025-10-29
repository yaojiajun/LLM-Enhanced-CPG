import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score matrix with noise addition
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.9  # Increase randomness

    # Modify the demand-based heuristic score matrix with additional noise and scaling
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.7 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3  # Adjust weights

    # Introduce perturbation for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5  # Increase noise level

    # Combine modified heuristic scores with different strategies for exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep the logic for other parts unchanged

    return cvrp_scores