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

    # Determine feasibility based on load, time windows, and current length constraints
    capacity_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * \
                           (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility assessment
    time_window_feasibility = ((arrival_times.unsqueeze(1) + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float() * \
                               ((arrival_times.unsqueeze(1) + current_distance_matrix + delivery_node_demands.unsqueeze(0) < time_windows[:, 0].unsqueeze(0)).float() + 1)

    # Length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Calculate heuristic scores: 
    # Combine distance with feasibility assessments
    heuristic_scores = (1 / (1 + current_distance_matrix)) * capacity_feasibility * time_window_feasibility * length_feasibility

    # Introduce randomness to avoid local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.01  # Small random perturbation
    heuristic_scores += random_noise

    # Normalize scores to range between -1 and 1
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min()) * 2 - 1

    return heuristic_scores