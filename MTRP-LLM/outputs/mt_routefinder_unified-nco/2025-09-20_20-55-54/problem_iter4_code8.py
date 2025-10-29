import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    EPSILON = 1e-8

    # Apply problem-specific constraints
    remaining_dist_score = 1 / (current_distance_matrix + EPSILON)
    
    # Add controlled randomness for exploration
    heuristic_scores = remaining_dist_score + torch.rand_like(current_distance_matrix) * 0.1

    return heuristic_scores