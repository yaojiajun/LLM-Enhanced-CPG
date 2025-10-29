import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    N_plus_1 = current_distance_matrix.size(1)
    pomo_size = current_distance_matrix.size(0)
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros((pomo_size, N_plus_1), device=current_distance_matrix.device)
    
    # Proximity score: inverse of distance (closer distances yield higher scores)
    proximity_scores = 1 / (1 + current_distance_matrix)
    
    # Load feasibility score: based on whether remaining load can accommodate delivery
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Time window feasibility score: based on arrival times and allowed time windows
    time_window_feasibility = ((arrival_times <= time_windows[:, 1].unsqueeze(0)).float() *
                                (arrival_times >= time_windows[:, 0].unsqueeze(0)).float())
    
    # Length feasibility score: based on remaining route duration
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine scores with weights
    heuristic_scores += proximity_scores * 0.5  # Weight for proximity
    heuristic_scores += load_feasibility * 0.3    # Weight for load feasibility
    heuristic_scores += time_window_feasibility * 0.15  # Weight for time window feasibility
    heuristic_scores += length_feasibility * 0.05  # Weight for length feasibility
    
    # Introduce randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Small random noise
    heuristic_scores += randomness

    return heuristic_scores