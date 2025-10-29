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
    
    # Calculate the total demand including deliveries and pickups
    total_demands = delivery_node_demands + pickup_node_demands
    
    # Create masks for load capacity, time windows and current length constraints
    capacity_mask = (current_load.unsqueeze(1) >= total_demands.unsqueeze(0)).float()
    current_length_mask = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Calculate time window penalties
    earliest_arrival = arrival_times + current_distance_matrix
    time_window_penalty_start = (earliest_arrival < time_windows[:, 0]).float() * (time_windows[:, 0] - earliest_arrival).clamp(min=0)
    time_window_penalty_end = (earliest_arrival > time_windows[:, 1]).float() * (earliest_arrival - time_windows[:, 1]).clamp(min=0)
    
    time_window_penalty = time_window_penalty_start + time_window_penalty_end
    
    # Heuristic score calculation
    # Start with distance matrix and adjust based on capacities and time window penalties
    heuristic_scores = -current_distance_matrix + 1000 * (1 - capacity_mask) + 1000 * (1 - current_length_mask) + time_window_penalty

    # Introduce stochasticity to encourage exploration
    randomness = torch.rand_like(heuristic_scores) * 10  # Randomness to avoid local optima
    heuristic_scores += randomness
    
    return heuristic_scores