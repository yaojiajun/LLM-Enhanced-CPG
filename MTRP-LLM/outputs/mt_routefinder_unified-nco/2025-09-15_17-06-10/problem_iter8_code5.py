import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Generate random weights for exploration
    rand_weights = torch.rand_like(current_distance_matrix)
    
    # Normalize the distance matrix
    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / max_distance
    
    # Calculate heuristic scores using various transformations
    score1 = torch.tanh(normalized_distance) * rand_weights
    score2 = torch.relu(torch.log1p(current_distance_matrix)) - rand_weights
    score3 = torch.sigmoid(current_load.unsqueeze(1).float() - delivery_node_demands.unsqueeze(0).float()) * rand_weights
    score4 = torch.relu(current_length.unsqueeze(1).float() - current_distance_matrix) + rand_weights
    
    # Combine scores for a rich heuristic indicator
    heuristic_scores = score1 + score2 + score3 - score4
    
    # Introduce adaptive randomness to avoid local optima
    randomness_adaptor = 0.1 * torch.rand_like(heuristic_scores)
    heuristic_scores += randomness_adaptor

    return heuristic_scores