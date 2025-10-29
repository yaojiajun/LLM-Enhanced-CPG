import torch
import torch

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Enhance heuristics based on node-specific constraints and balanced randomness
    # Implement your improved heuristic logic here

    heuristic_scores = torch.rand_like(current_distance_matrix)  # Placeholder for enhanced heuristic scores

    return heuristic_scores