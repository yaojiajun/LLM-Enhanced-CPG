import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Combine multiple indicators with enhanced randomness and non-linear transformations
    rand_factor = torch.rand_like(current_distance_matrix)
    score1 = torch.sigmoid(current_distance_matrix) * torch.cos(current_distance_matrix) * rand_factor
    score2 = torch.relu(torch.sin(current_distance_matrix)) * torch.tanh(current_distance_matrix) * (1 - rand_factor)
    
    # Apply activation functions and ensure normalization
    heuristic_scores = (score1 + score2) / 2.0

    return heuristic_scores