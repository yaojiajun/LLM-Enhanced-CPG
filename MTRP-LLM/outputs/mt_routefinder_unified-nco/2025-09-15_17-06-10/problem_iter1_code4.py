import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Calculate load feasibility matrix
    load_feasibility = torch.where(delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1), torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))

    # Calculate open load feasibility matrix
    open_load_feasibility = torch.where(delivery_node_demands_open.unsqueeze(0) <= current_load_open.unsqueeze(1), torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))

    # Calculate time window feasibility matrix
    time_window_feasibility = torch.where((arrival_times >= time_windows[:, 0].unsqueeze(0)) & (arrival_times <= time_windows[:, 1].unsqueeze(0)), torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))

    # Calculate pickup feasibility matrix
    pickup_feasibility = torch.where(pickup_node_demands.unsqueeze(0) <= current_load.unsqueeze(1), torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))

    # Calculate length feasibility matrix
    length_feasibility = torch.where(current_length.unsqueeze(1) >= 0, torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))

    # Combine feasibility matrices
    feasibility_matrix = load_feasibility * open_load_feasibility * time_window_feasibility * pickup_feasibility * length_feasibility

    # Calculate heuristic score based on inverse of current distance matrix
    heuristic_score = 1.0 / (current_distance_matrix + 1e-6)

    # Apply feasibility mask to heuristic score
    heuristic_score *= feasibility_matrix

    return heuristic_score