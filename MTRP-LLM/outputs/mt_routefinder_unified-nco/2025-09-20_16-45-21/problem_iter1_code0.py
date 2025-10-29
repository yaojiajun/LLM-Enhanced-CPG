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
    epsilon = 1e-8

    # Compute capacity constraints
    capacity_constraints = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Compute time window constraints
    time_constraints = ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)).float() * \
                       ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float()

    # Compute route duration constraints
    duration_constraints = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Heuristic scoring based on feasibility
    feasibility_scores = capacity_constraints * time_constraints * duration_constraints

    # Calculate heuristic score (lower distance is better)
    heuristic_scores = (1.0 / (current_distance_matrix + epsilon)) * feasibility_scores
    
    # Apply clamping to ensure only finite values
    heuristic_scores = torch.clamp(heuristic_scores, min=-1e5, max=1e5)

    # Apply controlled randomness to avoid local optima
    randomness = (torch.rand_like(heuristic_scores) * 0.01) - 0.005
    heuristic_scores += randomness

    # Ensure outputs are finite, replacing any inf values with a negative score
    heuristic_scores[~torch.isfinite(heuristic_scores)] = -1e5

    return heuristic_scores