import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Generate random noise matrix to enhance randomness
    noise_matrix = torch.randn_like(current_distance_matrix) * 0.1
    
    # Combine random noise with heuristic indicators
    heuristic_indicators = torch.rand_like(current_distance_matrix)
    heuristic_scores = heuristic_indicators + noise_matrix
    
    return heuristic_scores