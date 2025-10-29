import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    normalized_distance_scores = -current_distance_matrix * 1.2 + torch.randn_like(
        current_distance_matrix) * 0.6

    demand_scores = (2*current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 1.0 + torch.min(
        delivery_node_demands) / 2 - torch.randn_like(current_distance_matrix) * 0.3

    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Rest of the code remains the same as heuristics_v1

    return cvrp_scores