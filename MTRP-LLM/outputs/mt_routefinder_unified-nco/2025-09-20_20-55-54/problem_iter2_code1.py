import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Incorporate problem-specific constraints into score calculations
    norm_distance_matrix = current_distance_matrix / (current_distance_matrix.max() + 1e-8)
    noise = torch.rand_like(norm_distance_matrix) * 0.1
    norm_distance_matrix = norm_distance_matrix + noise
    
    # Apply heuristics with problem-specific constraints
    heuristic_scores = norm_distance_matrix * 0.8  # Example heuristic score calculation
    
    return heuristic_scores