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

    # Calculate total demand for each node
    total_demands = delivery_node_demands + pickup_node_demands
    
    # Calculate capacity feasibility for delivery nodes
    capacity_feasibility = (current_load.unsqueeze(1) >= total_demands.unsqueeze(0)).float()
    
    # Calculate time window feasibility
    time_window_feasibility = ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float() * \
                              ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)).float()
    
    # Calculate remaining length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine feasibility matrices
    feasibility_matrix = capacity_feasibility * time_window_feasibility * length_feasibility
    
    # Calculate heuristic scores based on distance, adjust for feasibility
    heuristic_scores = feasibility_matrix * (1.0 / (current_distance_matrix + 1e-6))  # Avoid division by zero
    
    # Introduce randomness to avoid local optima
    random_factor = torch.rand_like(heuristic_scores) * 0.1  # Noise to scores
    heuristic_scores += random_factor
    
    # Normalize scores to the range [0, 1]
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min() + 1e-6)

    return heuristic_scores