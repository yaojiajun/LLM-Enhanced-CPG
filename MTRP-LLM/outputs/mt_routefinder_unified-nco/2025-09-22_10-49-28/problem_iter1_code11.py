import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Adjust the normalized distance-based heuristic score matrix with increased randomness variation
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5  # Increased randomness variation

    # Adjust the demand-based heuristic score matrix with a focus on balancing load and randomness
    demand_scores = (current_load_open.unsqueeze(1) - delivery_node_demands_open) * 0.5 + torch.max(
        delivery_node_demands_open) / 2 + torch.randn_like(current_distance_matrix) * 0.3  # Balanced load focus with randomness

    # Introduce diversity for exploration with higher noise levels
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5  # Increased noise levels for exploration

    # Combine the updated heuristic scores for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remainder of the function remains unchanged from the original version 

    return cvrp_scores