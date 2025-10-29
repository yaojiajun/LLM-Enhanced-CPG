import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modified distance-based heuristic score matrix with increased randomness
    modified_distance_scores = torch.sqrt(current_distance_matrix) / torch.max(torch.sqrt(current_distance_matrix)) + torch.randn_like(
        current_distance_matrix) * 1.0

    # Adjusted delivery-based heuristic score matrix with diversified weightings
    adjusted_delivery_scores = (delivery_node_demands.unsqueeze(0) - 0.5 * current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce controlled randomness for exploration with moderate noise levels
    controlled_noise = torch.randn_like(current_distance_matrix) * 1.0

    # Combine the modified heuristic scores with strategic variations for balanced exploration
    final_scores = modified_distance_scores + adjusted_delivery_scores + controlled_noise
    
    return final_scores