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
    num_nodes = current_distance_matrix.shape[1]
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Compute potential capacity constraints
    delivery_capacity_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    pickup_capacity_feasible = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()

    # Compute time window feasibility
    time_window_min_feasible = (arrival_times.unsqueeze(2) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0).unsqueeze(0)).float()
    time_window_max_feasible = (arrival_times.unsqueeze(2) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0).unsqueeze(0)).float()
    time_window_feasible = time_window_min_feasible * time_window_max_feasible

    # Check remaining length constraints
    length_feasible = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine feasibility conditions
    feasible_conditions = delivery_capacity_feasible * pickup_capacity_feasible * time_window_feasible * length_feasible
    
    # Assign scores based on feasibility and distance
    heuristic_scores = feasible_conditions * (-current_distance_matrix)  # Encourage shorter paths

    # Introduce an element of randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Small random perturbation
    heuristic_scores += randomness

    return heuristic_scores