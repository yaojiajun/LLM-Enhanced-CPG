import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Generate random weights for exploration
    rand_weights = torch.rand_like(current_distance_matrix) * 0.5 + 0.5  # Random weights between 0.5 and 1.0 for better exploration
    # Normalize distances with respect to their maximum to compute relative distances
    max_distance = torch.max(current_distance_matrix, dim=1, keepdim=True)[0]
    normalized_distance = current_distance_matrix / max_distance
    
    # Heuristic calculations with various nonlinear transformations
    capacity_factor = (current_load.unsqueeze(1) - delivery_node_demands).clamp(min=0)
    length_factor = (current_length.unsqueeze(1) - current_distance_matrix).clamp(min=0)
    
    # Apply activation functions
    score1 = torch.sigmoid(normalized_distance) * rand_weights  # Nonlinear transformation using sigmoid
    score2 = torch.relu(capacity_factor) * torch.sigmoid(length_factor) + rand_weights  # Capacity and length impacts
    score3 = torch.tanh(time_windows[:, 0] - arrival_times)  # Time window consideration
    
    # Combine the scores, introducing randomness
    heuristic_scores = score1 + score2 + score3 * rand_weights
    
    # Normalize heuristic scores for consistent evaluation
    heuristic_scores = (heuristic_scores - heuristic_scores.min(dim=1, keepdim=True)[0]) / \
                       (heuristic_scores.max(dim=1, keepdim=True)[0] - heuristic_scores.min(dim=1, keepdim=True)[0] + 1e-6)
    
    return heuristic_scores