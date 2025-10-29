import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Define penalties and rewards
    penalty_for_exceeding_capacity = 1e5
    penalty_for_unmet_time_window = 1e5
    length_limit_penalty = 1e3
    
    # Initialize the score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)
    
    # Check for capacities
    feasible_capacity = (current_load.view(-1, 1) - delivery_node_demands.view(1, -1)) >= 0
    feasible_capacity_open = (current_load_open.view(-1, 1) - delivery_node_demands_open.view(1, -1)) >= 0
    
    # Combine feasibility
    combined_capacity = feasible_capacity & feasible_capacity_open
    
    # Assessing time window feasibility
    current_time_window_start = arrival_times
    time_window_open_start = time_windows[:, 0].view(1, -1)
    time_window_open_end = time_windows[:, 1].view(1, -1)
    
    feasible_time_window = (current_time_window_start <= time_window_open_end) & (current_time_window_start >= time_window_open_start)
    
    # Compute penalties based on current constraints
    score_matrix[~combined_capacity] -= penalty_for_exceeding_capacity
    score_matrix[~feasible_time_window] -= penalty_for_unmet_time_window

    # Evaluate distance if within feasible routes based on remaining length
    remaining_distance = current_length.view(-1, 1) - current_distance_matrix
    valid_route_indices = remaining_distance >= 0
    
    score_matrix[valid_route_indices] += (remaining_distance[valid_route_indices] + 1)  # give score based on remaining allowance

    # Introduce randomness to help escape local minima
    randomness = torch.randn_like(score_matrix) * 0.01
    score_matrix += randomness
    
    return score_matrix