import torch
def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    
    # Incorporate a combination of random weights with distance-based weights for heuristic evaluation
    heuristic_scores = torch.rand_like(current_distance_matrix) * 0.1 - current_distance_matrix * 0.9
    
    return heuristic_scores