import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    capacity_scores = torch.where(
        (delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)) &
        (pickup_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)),
        torch.zeros_like(current_distance_matrix),
        -torch.ones_like(current_distance_matrix)
    )
    
    duration_scores = torch.where(
        current_length.unsqueeze(1) >= current_distance_matrix,
        torch.zeros_like(current_distance_matrix),
        -torch.ones_like(current_distance_matrix)
    )
    
    time_window_scores = torch.where(
        (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) &
        (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)),
        torch.zeros_like(current_distance_matrix),
        -torch.ones_like(current_distance_matrix)
    )
    
    heuristic_scores = capacity_scores + duration_scores + time_window_scores
    
    return heuristic_scores