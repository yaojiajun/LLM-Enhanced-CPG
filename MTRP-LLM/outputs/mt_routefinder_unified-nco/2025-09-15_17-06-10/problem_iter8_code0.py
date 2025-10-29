import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    p, n_plus_one = current_distance_matrix.shape
    
    # Generate adaptive random weights for enhanced randomness
    rand_weights = torch.rand(p, n_plus_one)
    
    # Normalize distances while adding a small epsilon to prevent division by zero
    epsilon = 1e-6
    max_distance = torch.max(current_distance_matrix, dim=1, keepdim=True)[0] + epsilon
    normalized_distance = current_distance_matrix / max_distance
    
    # Score Calculation
    sigmoid_score = torch.sigmoid(normalized_distance) * rand_weights
    tanh_score = torch.tanh(current_distance_matrix) + rand_weights
    
    # Incorporate delivery and pickup demands in the scoring
    demand_penalty = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * -1.0
    pickup_penalty = (pickup_node_demands.unsqueeze(0) > current_load_open.unsqueeze(1)).float() * -1.0
    
    # Time window feasibility score
    current_time = arrival_times + normalized_distance
    time_window_score = ((current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Combine all components to create the heuristic score matrix
    heuristic_scores = (
        sigmoid_score 
        - tanh_score 
        + demand_penalty 
        + pickup_penalty 
        + time_window_score
    )
    
    return heuristic_scores