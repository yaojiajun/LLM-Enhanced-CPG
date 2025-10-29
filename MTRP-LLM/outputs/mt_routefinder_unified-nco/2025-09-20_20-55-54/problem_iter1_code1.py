import torch
import numpy as np
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Normalize input tensors
    delivery_node_demands_norm = delivery_node_demands / (torch.abs(delivery_node_demands).max() + 1e-8)
    current_load_norm = current_load / (torch.abs(current_load).max() + 1e-8)
    delivery_node_demands_open_norm = delivery_node_demands_open / (torch.abs(delivery_node_demands_open).max() + 1e-8)
    current_load_open_norm = current_load_open / (torch.abs(current_load_open).max() + 1e-8)

    # Compute heuristics scores
    score_matrix = current_distance_matrix * 0.5  # Example trivial scoring

    # Add controlled randomness
    random_noise = torch.randn_like(score_matrix) * 0.1
    score_matrix += random_noise

    # Clamp scores to avoid invalid values
    score_matrix = torch.clamp(score_matrix, -1e3, 1e3)

    return score_matrix