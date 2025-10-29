import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Enhanced heuristic computation with sophisticated randomness
    heuristic_scores = torch.rand_like(current_distance_matrix) * torch.randn_like(current_distance_matrix)  # Introducing enhanced randomness
    
    return heuristic_scores