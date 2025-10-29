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
    
    # Initialize the score matrix with zeros
    scores = torch.zeros_like(current_distance_matrix)

    # Establish thresholds and parameters
    capacity_threshold = 1e-6
    time_window_penalty = 10.0
    length_penalty = 5.0
    random_noise = torch.rand_like(scores) * 0.1

    # Check for capacity constraints
    capacity_surplus = (current_load.unsqueeze(1) >= delivery_node_demands[None, :]).float()
    capacity_surplus_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open[None, :]).float()

    # Evaluate time window feasibility
    time_window_validity = ((arrival_times < time_windows[:, 1].unsqueeze(0)) & 
                            (arrival_times + current_distance_matrix < time_windows[:, 0].unsqueeze(0))).float()
    
    # Compute penalties for time window violations
    time_window_score = (1 - time_window_validity) * time_window_penalty

    # Check duration limits
    length_validity = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Compute penalties for exceeding length budgets
    length_score = (1 - length_validity) * length_penalty

    # Combine scores: a higher score indicates a more favorable edge
    scores += (capacity_surplus + capacity_surplus_open) * 0.5  # Positive for feasible delivery routes
    scores -= time_window_score  # Penalize for time window issues
    scores -= length_score  # Penalize for exceeding length budgets
    
    # Add randomness to scores to encourage exploration
    scores += random_noise

    return scores