import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Improved heuristics algorithm with enhanced randomness and adaptability
    heuristics_scores = torch.empty_like(current_distance_matrix).uniform_(-1, 1)  # Improved random scores range [-1, 1]
    
    return heuristics_scores