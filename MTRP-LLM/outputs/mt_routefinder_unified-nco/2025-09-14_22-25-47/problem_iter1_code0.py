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
    
    pomo_size, N_plus_1 = current_distance_matrix.shape
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate available capacity
    available_capacity = current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)
    available_capacity_open = current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)
    
    # Calculate time window feasibility
    time_windows_start = time_windows[:, 0].unsqueeze(0)
    time_windows_end = time_windows[:, 1].unsqueeze(0)
    feasible_time = (arrival_times.unsqueeze(2) + current_distance_matrix >= time_windows_start) & \
                    (arrival_times.unsqueeze(2) + current_distance_matrix <= time_windows_end)

    # Calculate remaining budget for each route
    remaining_length = current_length.unsqueeze(1) >= current_distance_matrix
    
    # Compute scores
    score_based_on_capacity = available_capacity.float() * available_capacity_open.float() * 10
    score_based_on_time = feasible_time.float() * 5
    score_based_on_length = remaining_length.float() * 15
    
    # Combine scores
    heuristic_scores += score_based_on_capacity + score_based_on_time + score_based_on_length
    
    # Add randomness for enhanced exploration
    randomness = torch.rand_like(heuristic_scores) * 2 - 1  # Random values in range [-1, 1]
    heuristic_scores += randomness * 0.5  # Adjust randomness magnitude
    
    # Making sure that undesirable edges get a negative impact
    undesirable_edges_mask = torch.logical_not((available_capacity & available_capacity_open & feasible_time & remaining_length))
    heuristic_scores[undesirable_edges_mask] -= 20  # Penalty for undesirable edges
    
    return heuristic_scores