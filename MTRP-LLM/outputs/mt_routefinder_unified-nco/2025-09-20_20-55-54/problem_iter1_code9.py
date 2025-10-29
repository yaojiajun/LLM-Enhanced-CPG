import torch
import numpy as np
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize input tensors to prevent inf or -inf during divisions
    delivery_node_demands_norm = delivery_node_demands + 1e-8
    current_load_norm = current_load + 1e-8
    delivery_node_demands_open_norm = delivery_node_demands_open + 1e-8
    current_load_open_norm = current_load_open + 1e-8
    
    # Dividing by normalized tensors to avoid unstable divisions
    heuristic_score = torch.zeros_like(current_distance_matrix)
    # Add controlled randomness by adding small random noise
    noise = torch.rand_like(current_distance_matrix) * 1e-5
    
    # Compute heuristic based on normalized parameters and add controlled randomness
    heuristic_score = current_distance_matrix / (delivery_node_demands_norm.unsqueeze(0) + current_load_norm.unsqueeze(1)) + noise
    heuristic_score += current_distance_matrix / (delivery_node_demands_open_norm.unsqueeze(0) + current_load_open_norm.unsqueeze(1)) + noise
    
    return heuristic_score