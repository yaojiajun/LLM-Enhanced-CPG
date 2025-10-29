import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modified heuristics for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    
    # Compute a modified version of the distance-based heuristic score matrix with added controlled randomness
    normalized_distance_scores = -torch.sqrt(current_distance_matrix) / torch.max(current_distance_matrix) + torch.randn_like(current_distance_matrix) * 0.7

    # Compute a modified version of the demand-based heuristic score matrix with increased emphasis on delivery demand and randomness
    demand_scores = (2 * (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1))) / (torch.max(delivery_node_demands) + 1)  + torch.max(delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce controlled randomness with diversified strategies
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    total_scores = normalized_distance_scores + demand_scores + enhanced_noise
      

    return total_scores