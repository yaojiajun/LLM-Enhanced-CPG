import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Incorporate information from time windows, arrival times, and pickup demand
    time_diff = arrival_times[:, 1:] - arrival_times[:, :-1]  # Time differences between consecutive nodes
    time_window_diff = time_windows[:, 1:] - arrival_times[:, :-1]  # Time window differences between arrival times and time windows
    
    # Normalize tensors to prevent numerical instability
    epsilon = 1e-8
    time_diff_norm = time_diff + epsilon
    time_window_diff_norm = time_window_diff + epsilon

    # Calculate heuristic score matrix with controlled randomness and incorporating time and pickup demand information
    heuristic_scores = torch.rand_like(current_distance_matrix) + time_diff_norm / time_window_diff_norm + pickup_node_demands[1:]

    return heuristic_scores