import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Example modification: Inverse of current_distance_matrix for heuristics
    
    inverse_distance_matrix = 1.0 / (current_distance_matrix + 1e-8)  # Avoid division by zero
    
    delivery_score = delivery_node_demands / (current_load + 1e-8)  # Compute delivery score
    pickup_score = pickup_node_demands / (current_load + 1e-8)  # Compute pickup score
    
    total_score = inverse_distance_matrix + delivery_score - pickup_score
    
    return total_score