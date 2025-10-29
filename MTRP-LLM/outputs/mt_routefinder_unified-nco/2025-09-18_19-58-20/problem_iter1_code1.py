import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Capacity constraints checks
    valid_delivery_capacity = (delivery_node_demands.unsqueeze(0) <= current_load.view(-1, 1)).float()
    valid_delivery_capacity_open = (delivery_node_demands_open.unsqueeze(0) <= current_load_open.view(-1, 1)).float()
    
    # Time window feasibility checks
    time_windows_start = time_windows[:, 0]
    time_windows_end = time_windows[:, 1]
    arrival_time_tensor = arrival_times + current_distance_matrix
    valid_time_windows = ((arrival_time_tensor >= time_windows_start.unsqueeze(0)) & (arrival_time_tensor <= time_windows_end.unsqueeze(0))).float()
    
    # Duration limits checks
    valid_duration = (current_length.view(-1, 1) >= current_distance_matrix).float()

    # Calculate scores based on valid routes
    validity_matrix = valid_delivery_capacity * valid_delivery_capacity_open * valid_time_windows * valid_duration
    
    # Weights definition based on edge length for distance minimization
    distance_weights = 1 / (1 + current_distance_matrix)

    # Define a scoring mechanism that penalizes or rewards edges
    heuristic_scores = validity_matrix * distance_weights
    
    # Introduce randomness to scores to avoid local optima
    randomness_factor = torch.rand_like(heuristic_scores) * 0.1  # 10% randomness
    heuristic_scores += randomness_factor
    
    return heuristic_scores