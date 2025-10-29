import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Constants
    epsilon = 1e-8
    exploration_factor = 0.1
    exploitation_factor = 0.9
    
    # Normalize input tensors to prevent numerical instability and focus on valid routes
    delivery_node_demands_norm = delivery_node_demands + epsilon
    current_load_norm = current_load + epsilon
    current_length_norm = current_length + epsilon
    
    # Calculate feasible region for deliveries taking into account capacity and time windows
    capacity_constraints = (current_load_norm.unsqueeze(1) >= delivery_node_demands_norm).float()  # (pomo_size, N+1)
    time_window_constraints = (arrival_times <= time_windows[:, 1].unsqueeze(0)).float() * (arrival_times >= time_windows[:, 0].unsqueeze(0)).float()  # (pomo_size, N+1)

    # Combine constraints to filter feasible routes
    feasibility_matrix = capacity_constraints * time_window_constraints  # (pomo_size, N+1)

    # Add noise for exploration
    noise = torch.rand_like(current_distance_matrix) * exploration_factor * 2 - exploration_factor  # Random noise for exploration 

    # Compute heuristic scores while adhering to constraints
    heuristic_scores = exploitation_factor * (feasibility_matrix * (1 / (current_distance_matrix + epsilon))) + noise

    # Clamp scores to avoid unbounded values
    heuristic_scores = torch.clamp(heuristic_scores, min=-1.0, max=1.0)

    return heuristic_scores