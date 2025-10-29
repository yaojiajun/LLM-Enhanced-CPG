import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize input tensors with dynamic ranges
    delivery_node_demands_norm = delivery_node_demands / (delivery_node_demands.max(dim=0)[0] + 1e-8)
    current_load_norm = current_load / (current_load.max() + 1e-8)
    delivery_node_demands_open_norm = delivery_node_demands_open / (delivery_node_demands_open.max(dim=0)[0] + 1e-8)
    current_load_open_norm = current_load_open / (current_load_open.max() + 1e-8)

    # Compute heuristics scores
    score_matrix = current_distance_matrix * 0.5  # Example trivial scoring

    # Add controlled randomness
    random_noise = torch.randn_like(score_matrix) * 0.1
    score_matrix += random_noise

    # Clamp scores to avoid invalid values
    score_matrix = torch.clamp(score_matrix, -1e3, 1e3)

    return score_matrix