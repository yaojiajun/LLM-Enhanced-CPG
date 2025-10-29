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
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate feasibility masks for deliveries and pickups
    delivery_feasibility = (current_load.view(-1, 1) >= delivery_node_demands.unsqueeze(0)) & \
                           (current_length.view(-1, 1) >= current_distance_matrix)
    
    pickup_feasibility = (current_load_open.view(-1, 1) >= pickup_node_demands.unsqueeze(0)) & \
                         (current_length.view(-1, 1) >= current_distance_matrix)
    
    # Calculate time window feasibility
    time_window_mask = (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) & \
                       (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0))
    
    # Combine feasibility into a single mask
    feasibility_mask = delivery_feasibility & time_window_mask & pickup_feasibility
    
    # Calculate distance scores (shorter distances are better)
    distance_scores = -current_distance_matrix * feasibility_mask.float()
    
    # Introduce randomness based on constraint severity (adaptively scaling)
    randomness_scale = torch.pow((1 - feasibility_mask.float()), 2)  # Square to enhance effects on invalid options
    randomization = torch.rand_like(current_distance_matrix) * randomness_scale
    
    # Combine deterministic distance scores with randomization
    heuristic_scores = distance_scores + randomization
    
    # Normalize scores for stability
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min() + 1e-10)

    return heuristic_scores