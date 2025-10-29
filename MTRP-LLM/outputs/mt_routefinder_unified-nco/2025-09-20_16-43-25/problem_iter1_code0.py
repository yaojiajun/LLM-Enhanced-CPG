import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Initialize a heuristic score matrix with zeros
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Check for vehicle capacity constraints
    capacity_constraints = (current_load.unsqueeze(1) >= delivery_node_demands).float()
    capacity_constraints_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open).float()

    # Check for time window constraints
    time_window_start = (arrival_times < time_windows[:, 0].unsqueeze(0))
    time_window_end = (arrival_times > time_windows[:, 1].unsqueeze(0))
    time_windows_valid = ~(time_window_start | time_window_end)

    # Check for route duration constraints
    duration_constraints = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Compute heuristics based on constraints
    feasible_moves = capacity_constraints * capacity_constraints_open * time_windows_valid.float() * duration_constraints

    # Score calculation: negative distance for feasible moves, zero otherwise
    heuristic_scores += feasible_moves * (-current_distance_matrix)

    # Introduce randomness to promote exploration and avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += randomness

    # Return the heuristic score matrix
    return heuristic_scores