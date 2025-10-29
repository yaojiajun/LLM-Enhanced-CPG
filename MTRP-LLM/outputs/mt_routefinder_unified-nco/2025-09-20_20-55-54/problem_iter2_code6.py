import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    EPSILON = 1e-8
    
    # Compute a heuristic score matrix with random perturbations
    heuristic_scores = 1 / (current_distance_matrix + EPSILON) + torch.rand_like(current_distance_matrix) * 0.1
    
    return heuristic_scores