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
    
    # Initialize score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)

    # Compute penalties for exceeding capacity
    delivery_capacity_penalty = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * 1e6
    pickup_capacity_penalty = (pickup_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * 1e6

    # Compute time window feasibility
    time_window_penalty = (
        (arrival_times.unsqueeze(1) + current_distance_matrix) < time_windows[:, 0].unsqueeze(0)
    ).float() * 1e6 + (
        (arrival_times.unsqueeze(1) + current_distance_matrix) > time_windows[:, 1].unsqueeze(0)
    ).float() * 1e6

    # Compute duration limit penalty
    duration_limit_penalty = (current_length.unsqueeze(1) < current_distance_matrix).float() * 1e6

    # Combine penalties to score matrix
    penalties = delivery_capacity_penalty + pickup_capacity_penalty + time_window_penalty + duration_limit_penalty

    # Compute heuristic score: inversely proportional to distances, adjusted by penalties
    score_matrix = -current_distance_matrix + penalties

    # Add randomness for diversity
    randomness = torch.rand_like(score_matrix) * 1e-2
    score_matrix += randomness

    return score_matrix