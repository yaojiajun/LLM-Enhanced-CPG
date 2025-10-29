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

    epsilon = 1e-8
    
    # Calculate feasibility based on load capacity
    feasible_load = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_load_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Calculate time window feasibility
    time_now = arrival_times + current_distance_matrix
    in_time_window = ((time_now >= time_windows[:, 0].unsqueeze(0)) & 
                      (time_now <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Calculate remaining time within the route
    remaining_time = current_length.unsqueeze(1) - current_distance_matrix
    
    # Ensure no negative time left
    valid_time = (remaining_time >= 0).float()
    
    # Scoring based on feasibility and distance
    base_scores = feasible_load * in_time_window * valid_time
    distance_scores = 1 / (current_distance_matrix + epsilon)
    
    # Combine scores with randomness
    controlled_randomness = torch.rand(current_distance_matrix.shape) * 0.1
    heuristic_scores = base_scores * distance_scores + controlled_randomness
    
    # Masking non-finite scores
    heuristic_scores[~torch.isfinite(heuristic_scores)] = 0
    
    return heuristic_scores