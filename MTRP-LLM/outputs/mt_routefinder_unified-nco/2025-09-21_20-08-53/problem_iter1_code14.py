import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    delivery_scores = (torch.sum(torch.exp(delivery_node_demands), dim=0) / 100) - torch.sum(current_load)

    pickup_scores = (torch.sum(torch.sqrt(torch.abs(current_load))) / 10) - torch.sum(pickup_node_demands)

    cvrp_scores = normalized_distance_scores + delivery_scores + pickup_scores
    
    # Keep the rest of the implementation unchanged
    
    return cvrp_scores