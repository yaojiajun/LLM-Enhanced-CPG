import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize distance matrix
    normalized_distance_matrix = current_distance_matrix / (current_distance_matrix.max() + 1e-8)
    
    # Calculate load ratios with epsilon for numerical stability
    load_ratios = current_load.unsqueeze(1) / (delivery_node_demands + 1e-8)
    
    # Calculate time window ratios with epsilon for numerical stability
    window_ratios = (arrival_times[:, 1:] - arrival_times[:, :-1]) / (time_windows[:, 1:, 1] - time_windows[:, 1:, 0] + 1e-8)
    
    # Use randomness for exploration
    random_scores = torch.rand_like(current_distance_matrix) * 0.1
    
    # Combine ratios and randomness to generate heuristic scores
    heuristic_scores = normalized_distance_matrix + load_ratios + window_ratios + random_scores
    
    return heuristic_scores