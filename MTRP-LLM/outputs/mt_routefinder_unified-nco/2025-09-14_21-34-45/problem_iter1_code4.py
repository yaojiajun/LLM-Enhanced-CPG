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
    
    # Constants
    epsilon = 1e-6  # Small value to avoid division by zero
    max_distance = torch.max(current_distance_matrix) + 1  # Max distance for normalization

    # Calculate available capacity for deliveries and pickups
    available_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    available_capacity_open = current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)
    
    # Check feasibility based on capacity constraints
    capacity_feasible_delivery = (available_capacity >= 0).float()  # Shape: (pomo_size, N+1)
    capacity_feasible_pickup = (available_capacity_open >= 0).float()  # Shape: (pomo_size, N+1)
    
    # Calculate effective service times considering time windows
    service_time_penalty = (arrival_times.unsqueeze(1) + current_distance_matrix > time_windows[:, 1].unsqueeze(0)).float()
    
    # Check if the arrival time fits within the time windows
    time_window_feasible = ((arrival_times.unsqueeze(1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)) & 
                             (arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Duration constraints
    duration_feasible = (current_length.unsqueeze(1) - current_distance_matrix >= 0).float()
    
    # Heuristic scores calculation
    score_matrix = (capacity_feasible_delivery * capacity_feasible_pickup * time_window_feasible * duration_feasible * 
                    (1 - service_time_penalty)) * (max_distance - current_distance_matrix) / max_distance
    
    # Introduce enhanced randomness to avoid local optima
    random_noise = torch.rand_like(score_matrix) * 0.1  # Small random noise
    score_matrix += random_noise
    
    return score_matrix