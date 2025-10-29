import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic indicators with random values to introduce exploration
    heuristic_indicators = torch.rand_like(current_distance_matrix)

    # Calculate the feasible deliveries based on remaining load and delivery demands
    feasible_deliveries = (current_load[:, None] >= delivery_node_demands) & (current_load_open[:, None] >= delivery_node_demands_open)

    # Calculate wait times and score modifications based on time windows
    current_time = arrival_times
    time_window_scores = torch.zeros_like(current_distance_matrix)
    
    for i in range(time_windows.shape[0]):
        early, late = time_windows[i]
        # If arrival time is too early, penalty for waiting
        wait_penalty = torch.clamp(early - current_time[:, i], min=0)
        # If arriving too late, penalty for exceeding time window
        lateness_penalty = torch.clamp(current_time[:, i] - late, min=0)
        time_window_scores[:, i] = - (wait_penalty + lateness_penalty)

    # Update heuristic indicators based on capacity, delivery demands, and time windows
    heuristic_indicators += feasible_deliveries.float() * (1.0 / (current_distance_matrix + 1e-6))  # Avoid division by zero
    heuristic_indicators += time_window_scores * 0.5  # Scale time window impact to heuristic
    
    # Add randomness for exploration
    noise = torch.normal(mean=0, std=0.1, size=heuristic_indicators.shape).to(heuristic_indicators.device)
    heuristic_indicators += noise
    
    # Final penalties for lengths, if applicable
    length_penalties = (current_length[:, None] < current_distance_matrix).float() * -1.0
    heuristic_indicators += length_penalties

    return heuristic_indicators