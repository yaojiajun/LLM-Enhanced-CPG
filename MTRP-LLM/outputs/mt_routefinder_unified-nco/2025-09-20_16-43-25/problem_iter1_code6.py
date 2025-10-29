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

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Capacity feasibility checks
    capacity_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    capacity_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility checks
    time_window_feasibility = ((arrival_times < time_windows[:, 1].unsqueeze(0)).float() * 
                              (arrival_times >= time_windows[:, 0].unsqueeze(0)).float())
    
    # Duration limits feasibility checks
    duration_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine feasibility checks
    feasibility_matrix = capacity_feasibility * capacity_feasibility_open * time_window_feasibility * duration_feasibility
    
    # Compute heuristic scores: shorter distance, penalizing infeasible moves
    heuristic_scores = feasibility_matrix * (-current_distance_matrix)

    # Introduce randomness to avoid convergence to local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.1  # Random noise to explore more options
    heuristic_scores += random_noise
    
    return heuristic_scores