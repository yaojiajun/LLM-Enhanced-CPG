import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Improved heuristic function with additional nonlinearity and randomness
    rand_weights = torch.rand_like(current_distance_matrix) * 2 - 1
    normalized_distance = current_distance_matrix / torch.max(current_distance_matrix)
    
    score1 = (torch.sigmoid(normalized_distance) + torch.tanh(torch.sin(current_distance_matrix))) * rand_weights
    score2 = torch.relu(torch.tanh(current_distance_matrix)) - rand_weights
    score3 = torch.sigmoid(torch.exp(-current_distance_matrix)) * rand_weights
    heuristic_scores = score1 - score2 + score3

    return heuristic_scores