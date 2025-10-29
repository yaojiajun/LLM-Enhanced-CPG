import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Normalize input tensors to prevent numerical instability
    epsilon = 1e-8
    delivery_node_demands_norm = delivery_node_demands + epsilon
    current_load_norm = current_load + epsilon
    delivery_node_demands_open_norm = delivery_node_demands_open + epsilon
    current_load_open_norm = current_load_open + epsilon

    # Calculate heuristic score matrix with controlled randomness and balanced exploration-exploitation
    exploration_factor = 0.1
    exploitation_factor = 0.9
    random_scores = torch.rand_like(current_distance_matrix) * (2 * exploration_factor) - exploration_factor
    heuristic_scores = random_scores + exploitation_factor * current_distance_matrix

    return heuristic_scores