import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Normalize distance matrix
    norm_distance_matrix = current_distance_matrix / (current_distance_matrix.max() + 1e-8)
    
    # Calculate load ratios
    load_ratios = current_load.unsqueeze(1) / (delivery_node_demands + 1e-8)
    
    # Penalize infeasible load ratios
    load_ratios = torch.where(load_ratios > 1, -load_ratios, load_ratios)
    
    # Incorporate time window constraints
    time_window_scores = (arrival_times - time_windows[:, 1].unsqueeze(0)) / (time_windows[:, 1].unsqueeze(0) - time_windows[:, 0].unsqueeze(0) + 1e-8)
    
    # Randomness for exploration
    randomness = torch.rand_like(current_distance_matrix) * 0.1
    
    heuristic_scores = norm_distance_matrix + load_ratios + time_window_scores + randomness
    
    return heuristic_scores