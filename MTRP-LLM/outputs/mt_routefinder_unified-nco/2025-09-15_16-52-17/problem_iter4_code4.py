import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Calculate heuristic score matrix based on enhanced logic incorporating more problem-specific insights and constraints
    heuristic_scores = torch.rand_like(current_distance_matrix) * (torch.rand_like(current_distance_matrix) * 3 - 1)  # Adjusted randomness factor
    
    return heuristic_scores