import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalize input tensors
    eps = 1e-8
    delivery_node_demands_norm = delivery_node_demands + eps
    current_load_norm = current_load + eps
    delivery_node_demands_open_norm = delivery_node_demands_open + eps
    current_load_open_norm = current_load_open + eps

    # Calculate heuristic scores based on problem-specific constraints and insights from prior heuristics
    heuristic_scores = torch.sqrt(current_distance_matrix / delivery_node_demands_norm.unsqueeze(0))  # Example heuristic score calculation

    return torch.clamp(heuristic_scores, min=0.0)  # Ensure only non-negative finite scores