import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):

    # Improved heuristic implementation
    noise_level = torch.rand(1) * 0.5  # Generate random noise level between 0 and 0.5
    heuristic_scores = torch.rand_like(current_distance_matrix) * 2 - 1 + noise_level

    return heuristic_scores