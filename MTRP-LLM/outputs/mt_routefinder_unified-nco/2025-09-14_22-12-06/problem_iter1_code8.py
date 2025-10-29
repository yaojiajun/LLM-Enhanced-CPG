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
    
    # Initialize the heuristic score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)

    # Calculate feasible visits based on load constraints
    feasible_deliveries = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) | \
                          (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Calculate time window feasibility
    time_window_feasibility = (arrival_times.unsqueeze(2) + current_distance_matrix.unsqueeze(1) >= 
                                time_windows[:, :, 0].unsqueeze(0).unsqueeze(0)) & \
                               (arrival_times.unsqueeze(2) + current_distance_matrix.unsqueeze(1) <= 
                                time_windows[:, :, 1].unsqueeze(0).unsqueeze(0))
    
    # Calculate remaining length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)

    # Combine feasibility constraints
    feasibility_mask = feasible_deliveries & time_window_feasibility & length_feasibility
    
    # Adjust scores for feasible routes, higher scores for shorter distances
    score_matrix[feasibility_mask] = -current_distance_matrix[feasibility_mask]
    
    # Penalty for infeasible routes (not meeting demands or time windows)
    score_matrix[~feasibility_mask] = float('inf')  # Assign high penalty for infeasible routes
    
    # Introduce randomness to avoid local optima
    random_noise = torch.rand_like(score_matrix) * 0.01  # small random noise
    score_matrix += random_noise

    return score_matrix