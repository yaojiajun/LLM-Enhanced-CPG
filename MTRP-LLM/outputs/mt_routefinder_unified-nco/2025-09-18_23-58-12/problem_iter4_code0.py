import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Enhanced heuristic function with adaptive weights and increased randomness diversity
    random_modifier = torch.randn_like(current_distance_matrix) * 0.2
    distance_weights = torch.rand_like(current_distance_matrix) * 0.5
    
    return current_distance_matrix + random_modifier + distance_weights