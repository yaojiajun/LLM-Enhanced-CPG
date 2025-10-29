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
    pomo_size, num_nodes = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)

    # Calculate potential delivery and pickup scores
    delivery_capacity = current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)
    pickup_capacity = current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)

    # Time window feasibility
    current_time = arrival_times + current_distance_matrix
    within_time_windows = (current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))

    # Duration limit feasibility
    feasible_duration = current_length.unsqueeze(1) >= current_distance_matrix

    # All feasibility checks
    feasible_to_visit = delivery_capacity & within_time_windows & feasible_duration

    # Scoring based on distance and feasibility
    distance_scores = torch.where(feasible_to_visit, -current_distance_matrix, float('inf'))

    # Introduce randomness to scores to prevent local optima
    randomness = torch.rand((pomo_size, num_nodes), device=current_distance_matrix.device) * 0.1
    heuristic_scores = distance_scores + randomness

    # Normalize scores to ensure it represents comparative utility
    heuristic_scores = (heuristic_scores - heuristic_scores.mean(dim=1, keepdim=True)) / \
                       (heuristic_scores.std(dim=1, keepdim=True) + 1e-5)

    return heuristic_scores