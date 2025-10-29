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

    # Initialize a heuristic score matrix
    heuristic_scores = -current_distance_matrix.clone()

    # Calculate scores based on delivery demands
    feasible_deliveries = (delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)) & (delivery_node_demands_open.unsqueeze(0) <= current_load_open.unsqueeze(1))
    heuristic_scores += feasible_deliveries.float() * 10.0  # Favorable scores for feasible deliveries

    # Adjust scores based on time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasibility = (current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))
    heuristic_scores += time_window_feasibility.float() * 5.0  # Moderate scores for time window feasibility

    # Penalize for violations of length constraints
    length_violation = (current_length.unsqueeze(1) < current_distance_matrix.sum(dim=0)).float() * -5.0
    heuristic_scores += length_violation

    # Incorporate randomness to avoid local optima
    noise = torch.rand_like(heuristic_scores) * 0.5  # Add random noise in the range [0, 0.5]
    heuristic_scores += noise

    # Ensure we don't get negative scores, making them range from 0 to high positive
    heuristic_scores = torch.clamp(heuristic_scores, min=0)

    return heuristic_scores