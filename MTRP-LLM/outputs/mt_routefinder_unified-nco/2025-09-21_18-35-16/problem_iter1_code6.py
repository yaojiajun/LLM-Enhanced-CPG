import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Distance-based heuristic score matrix modification
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5
    
    # Modified delivery-based heuristic score matrix with adjusted sensitivity and randomness for diversification
    delivery_scores = (torch.sqrt(delivery_node_demands) - torch.sqrt(current_load)) * 0.9 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5
    
    # Generate noise for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5
    
    # Combine different heuristic scores with varied exploration strategies
    cvrp_scores = normalized_distance_scores + delivery_scores + enhanced_noise

    # Keep the calculation of other scores unchanged
    
    return cvrp_scores