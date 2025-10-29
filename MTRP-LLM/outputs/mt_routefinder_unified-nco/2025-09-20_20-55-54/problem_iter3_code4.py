import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Adaptive scaling considering demand and distance
    scaled_distance = current_distance_matrix / (delivery_node_demands.unsqueeze(0) + 1e-8)
    
    # Incorporating time window constraints
    time_window_weights = 1 - ((arrival_times - time_windows[:,0].unsqueeze(0)) / (time_windows[:,1] - time_windows[:,0] + 1e-8))
    
    # Introducing randomness and informedness balance
    heuristic_scores = torch.rand_like(current_distance_matrix) * 0.5 + scaled_distance * 0.3 + time_window_weights * 0.2
    
    return heuristic_scores