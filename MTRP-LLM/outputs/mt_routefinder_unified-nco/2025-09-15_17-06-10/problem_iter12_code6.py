import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Generate random weights for exploration
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)
    rand_weights3 = torch.rand_like(current_distance_matrix)

    # Normalize distances with better stability and prevent division by zero
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix, dim=1, keepdim=True)[0] + 1e-10)

    # Score calculation using smoother activation functions
    score1 = torch.sin(normalized_distance * torch.pi) * rand_weights1
    score2 = torch.sigmoid(current_distance_matrix) * rand_weights2
    score3 = torch.tanh(delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1e-10)) * rand_weights3

    # Combined scores with balance between exploration and exploitation
    heuristic_scores = score1 - score2 + score3

    return heuristic_scores