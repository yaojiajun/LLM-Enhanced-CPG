import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    
    # Combine multiple indicators with randomness
    rand_factor = torch.rand_like(current_distance_matrix)
    score1 = torch.sigmoid(current_distance_matrix) * rand_factor
    score2 = torch.relu(torch.sin(current_distance_matrix)) * (1 - rand_factor)
    
    # Apply activation functions and ensure normalization
    heuristic_scores = (score1 + score2) / 2.0

    return heuristic_scores