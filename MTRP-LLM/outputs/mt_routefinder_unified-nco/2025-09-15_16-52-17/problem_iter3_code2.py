import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Constraints evaluation
    # Check if visiting nodes respects delivery capacity constraints
    delivery_capacity = (delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)).float()
    open_delivery_capacity = (delivery_node_demands_open.unsqueeze(0) <= current_load_open.unsqueeze(1)).float()
    
    # Check if visiting nodes respects current length constraints
    length_capacity = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Check time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasibility = (current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))
    
    # Combined constraints
    feasibility_filter = delivery_capacity * open_delivery_capacity * length_capacity * time_window_feasibility.float()
    
    # Heuristic score calculation
    score_distance = -current_distance_matrix * feasibility_filter  # Minimize distance while respecting constraints
    random_scores = torch.rand_like(current_distance_matrix) * 0.5  # Randomness introduced
    
    # Enhance scores based on feasibility
    heuristic_scores = score_distance + random_scores
    
    # Introduce adaptive penalties based on historical edge use (simulated by random noise for simplicity)
    adaptive_penalty = torch.rand_like(heuristic_scores) * 0.05 * (1 - feasibility_filter)  # Heavier penalization for infeasible routes
    heuristic_scores -= adaptive_penalty
    
    return heuristic_scores