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
    
    # Penalty for violating capacity constraints
    capacity_penalty = (delivery_node_demands > current_load.unsqueeze(1)).float() * 1e3  # High penalty for exceeding capacity
    capacity_penalty_open = (delivery_node_demands_open > current_load_open.unsqueeze(1)).float() * 1e3  # Open routes

    # Penalty for violating time windows
    earliest_arrivals = arrival_times + current_distance_matrix
    time_window_violation = (earliest_arrivals < time_windows[:, 0].unsqueeze(0)).float() * 1e3  # Early arrivals
    waiting_penalty = (earliest_arrivals > time_windows[:, 1].unsqueeze(0)).float() * (earliest_arrivals - time_windows[:, 1].unsqueeze(0))  # Late arrivals

    # Penalty for exceeding route length
    length_penalty = (current_length.unsqueeze(1) < current_distance_matrix).float() * 1e3  # If exceeding length constraints

    # Calculate heuristic scores
    heuristic_scores = -(current_distance_matrix - capacity_penalty - capacity_penalty_open - time_window_violation - length_penalty + waiting_penalty)
    
    # Incorporate randomness to avoid local optima
    randomness_factor = torch.randn_like(heuristic_scores) * 0.05  # Small amount of randomness
    heuristic_scores += randomness_factor

    # Ensure heuristic scores are non-negative for promising edges
    heuristic_scores = torch.clamp(heuristic_scores, min=-1e3, max=None)

    return heuristic_scores