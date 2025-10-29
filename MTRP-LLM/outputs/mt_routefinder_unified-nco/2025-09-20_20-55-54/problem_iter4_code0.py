import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Normalize input parameters
    delivery_node_demands_norm = delivery_node_demands / (torch.abs(delivery_node_demands).max() + 1e-8)
    current_load_norm = current_load / (torch.abs(current_load).max() + 1e-8)
    delivery_node_demands_open_norm = delivery_node_demands_open / (torch.abs(delivery_node_demands_open).max() + 1e-8)
    current_load_open_norm = current_load_open / (torch.abs(current_load_open).max() + 1e-8)

    # Introduce adaptive adjustments based on input interactions
    heuristic_scores = current_distance_matrix * 0.5  # Placeholder score computation

    # Add controlled randomness in score adjustments
    heuristic_scores += torch.randn_like(heuristic_scores) * 0.1

    # Clamp scores to avoid invalid values
    heuristic_scores = torch.clamp(heuristic_scores, -1e3, 1e3)

    return heuristic_scores