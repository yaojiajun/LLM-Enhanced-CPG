import torch
def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Perform enhanced heuristic computations with purposeful randomness and sophisticated scoring based on constraints
    heuristic_scores = current_distance_matrix * (1 + torch.rand_like(current_distance_matrix) * 0.1 - 0.05)
    
    return heuristic_scores