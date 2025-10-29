import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Generate random weights and biases for adaptive weighting
    rand_weights = torch.rand_like(current_distance_matrix)
    rand_bias = torch.rand(rand_weights.shape[0], 1, device=current_distance_matrix.device)

    # Normalize the distance matrix
    max_distance = current_distance_matrix.max(dim=1, keepdim=True)[0]
    normalized_distance = current_distance_matrix / max_distance
    
    # Calculate remaining demand scores
    delivery_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    pickup_capacity = current_load_open.unsqueeze(1) - pickup_node_demands.unsqueeze(0)
    
    # Calculate time window feasibility
    arrival_time = arrival_times - current_distance_matrix
    time_window_feasibility = (arrival_time >= time_windows[:, 0].unsqueeze(0)) & (arrival_time <= time_windows[:, 1].unsqueeze(0))
    
    # Create scores based on the different indicators
    load_score = torch.sigmoid(delivery_capacity) + torch.sigmoid(pickup_capacity)
    time_window_score = time_window_feasibility.float()
    
    # Apply non-linear transformations and combine them
    score1 = torch.tanh(normalized_distance) * rand_weights * load_score
    score2 = torch.relu(torch.cos(normalized_distance)) + rand_bias
    heuristic_scores = score1 + time_window_score - score2

    # Normalize heuristic scores
    heuristic_scores = (heuristic_scores - heuristic_scores.min(dim=1, keepdim=True)[0]) / (heuristic_scores.max(dim=1, keepdim=True)[0] - heuristic_scores.min(dim=1, keepdim=True)[0] + 1e-10)

    return heuristic_scores