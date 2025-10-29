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
    
    # Ensure tensor dimensions match requirements
    pomo_size, N_plus_one = current_distance_matrix.shape

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, N_plus_one), device=current_distance_matrix.device)

    # Check for load constraints
    load_constraints = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * \
                       (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Check for duration constraints
    duration_constraints = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Check for time window constraints
    time_window_feasibility = ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)).float() * \
                              ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float()

    # Compute base heuristic scores as the inverse of the distance
    base_scores = 1 / (current_distance_matrix + 1e-9)  # Avoid division by zero

    # Combine constraints, base scores into final heuristic scores
    heuristic_scores = base_scores * load_constraints * duration_constraints * time_window_feasibility

    # Introduce randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Random component to shake the scores

    # Final heuristic score with randomness included
    heuristic_scores += randomness

    return heuristic_scores