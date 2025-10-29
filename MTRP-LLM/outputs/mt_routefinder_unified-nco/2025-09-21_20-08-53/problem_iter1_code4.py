import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the distance-based heuristic score matrix with modified ranking based on distances
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7
    
    # Compute the demand-based heuristic score matrix with modified emphasis on high-demand nodes
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with modified noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with modified strategies for exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep other parts of the function unchanged
    
    return cvrp_scores