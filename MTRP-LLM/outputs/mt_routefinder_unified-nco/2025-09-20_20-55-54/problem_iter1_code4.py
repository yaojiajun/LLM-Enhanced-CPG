import torch
import numpy as np
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize inputs to prevent inf or -inf
    epsilon = 1e-8
    delivery_node_demands_normalized = delivery_node_demands + epsilon
    current_load_normalized = current_load + epsilon
    delivery_node_demands_open_normalized = delivery_node_demands_open + epsilon
    current_load_open_normalized = current_load_open + epsilon

    # Heuristics computation using controlled randomness and avoiding inf or -inf
    heuristic_scores = current_distance_matrix / (delivery_node_demands_normalized.unsqueeze(0) + current_load_normalized.unsqueeze(1) + epsilon)
    heuristic_scores -= current_distance_matrix / (delivery_node_demands_open_normalized.unsqueeze(0) + current_load_open_normalized.unsqueeze(1) + epsilon)
    
    # Clamp or mask invalid values
    heuristic_scores = torch.where(torch.isinf(heuristic_scores) | torch.isnan(heuristic_scores), torch.zeros_like(heuristic_scores), heuristic_scores)
    
    return heuristic_scores