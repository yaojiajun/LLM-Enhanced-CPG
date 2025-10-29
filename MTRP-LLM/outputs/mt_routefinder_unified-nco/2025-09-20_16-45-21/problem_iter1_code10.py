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
    
    # Epsilon for numerical stability
    epsilon = 1e-8

    # Compute the potential delivery feasibility
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()

    # Calculate the time window feasibility
    arrival_upper = arrival_times + current_distance_matrix
    arrival_lower = arrival_times + current_distance_matrix + time_windows[:, 0].unsqueeze(0)
    time_feasible = ((arrival_upper <= time_windows[:, 1].unsqueeze(0)) & 
                     (arrival_lower >= time_windows[:, 0].unsqueeze(0))).float()

    # Compute distance penalties with a controlled randomness term
    # This randomness helps to explore new routes to avoid local optima
    randomness = (torch.rand_like(current_distance_matrix) * 0.1) - 0.05
    distance_penalty = current_distance_matrix + randomness

    # Normalize distance with current remaining lengths while avoiding division by zero
    remaining_lengths = current_length.unsqueeze(1) + epsilon
    normalized_distances = distance_penalty / remaining_lengths

    # Combine scores: positive for feasible edges, negative for infeasible ones
    scores = feasible_delivery * time_feasible - normalized_distances

    # Mask out non-finite scores to keep them within bounds
    scores[~torch.isfinite(scores)] = -float('inf')

    return scores