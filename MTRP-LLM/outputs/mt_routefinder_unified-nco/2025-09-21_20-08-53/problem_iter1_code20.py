import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute a modified distance-based heuristic score matrix with additional noise and randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(current_distance_matrix) * 0.7

    # Compute the demand-based heuristic score matrix with altered emphasis and randomness
    delivery_score = (2*delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Combine the different heuristic scores with different weights and randomness for exploration
    cvrp_scores = normalized_distance_scores + delivery_score

    # Remaining parts remain the same as in the original function

    return cvrp_scores