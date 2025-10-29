import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # cvrp_scores
    # Modify the distance-based heuristic score matrix calculation for 'current_distance_matrix' with increased randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.9
    
    # Modify the demand-based heuristic score matrix calculation for 'delivery_node_demands' and 'current_load'
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 1.0 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(delivery_node_demands) * 0.6

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise
    
    # Retain logic for the remaining parts of the function
    
    return cvrp_scores