import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize distances to [0, 1]
    max_distance = current_distance_matrix.max(dim=1, keepdim=True)[0]
    normalized_distance = current_distance_matrix / (max_distance + 1e-6)

    # Compute soft penalties based on constraints
    demand_penalty = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * 10.0
    time_penalty = ((arrival_times + normalized_distance) > time_windows[:, 1].unsqueeze(0)).float() * 10.0
    length_penalty = (current_length.unsqueeze(1) < normalized_distance).float() * 10.0

    # Randomness for exploration
    rand_scores = torch.rand_like(current_distance_matrix) * 0.5  # Limited randomness to enhance exploration

    # Calculate raw scores
    base_scores = 1.0 - normalized_distance - demand_penalty - time_penalty - length_penalty + rand_scores

    # Apply a sigmoid to emphasize feasible routes
    heuristic_scores = torch.sigmoid(base_scores)

    # Normalize the scores to range [-1, 1]
    heuristic_scores = (heuristic_scores - 0.5) * 2

    return heuristic_scores