import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Generate random weights for adaptive scoring
    rand_weights = torch.rand_like(current_distance_matrix)
   
    # Normalize the distance matrix for better scaling
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-6)
    
    # Adaptive scoring based on multiple nonlinear transformations
    score1 = torch.sin(normalized_distance * torch.pi) * rand_weights
    score2 = torch.log1p(normalized_distance) + rand_weights * 0.5
    score3 = torch.tanh(normalized_distance * 1.5) - rand_weights
    score4 = torch.relu(torch.sqrt(normalized_distance)) * rand_weights

    # Combine the scores for a final heuristic score matrix
    heuristic_scores = (score1 + score3) - (score2 + score4)

    # Incorporate a randomness factor to increase diversity
    heuristic_scores += torch.rand_like(heuristic_scores) * 0.1

    return heuristic_scores