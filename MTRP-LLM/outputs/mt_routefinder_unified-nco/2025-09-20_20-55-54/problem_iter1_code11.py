import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize inputs with epsilon
    epsilon = 1e-8
    delivery_node_demands_open_norm = delivery_node_demands_open + epsilon
    current_load_open_norm = current_load_open + epsilon

    # Compute heuristic scores based on various features
    score = current_distance_matrix / delivery_node_demands.view(1, -1)  # Example heuristic rule based on distance and demand
    score += arrival_times / delivery_node_demands_open_norm.view(1, -1)  # Example heuristic rule based on arrival times and open route demands

    return score