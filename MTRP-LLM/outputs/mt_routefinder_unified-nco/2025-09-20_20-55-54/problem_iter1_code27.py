import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalize input tensors to prevent numerical instability
    epsilon = 1e-8
    delivery_node_demands_norm = delivery_node_demands + epsilon
    current_load_norm = current_load + epsilon
    delivery_node_demands_open_norm = delivery_node_demands_open + epsilon
    current_load_open_norm = current_load_open + epsilon

    # Calculate heuristic score matrix with controlled randomness
    heuristic_scores = torch.rand_like(current_distance_matrix) * 2 - 1

    return heuristic_scores