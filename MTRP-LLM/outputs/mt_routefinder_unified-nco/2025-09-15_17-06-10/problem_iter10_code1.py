import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix more robustly
    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / (max_distance + 1e-6)

    # Heuristic scores based on distance and capacity constraints
    capacity_penalty = torch.clamp((delivery_node_demands - current_load.unsqueeze(1)), min=0) # Positive if demand exceeds available load
    duration_penalty = torch.clamp(current_length.unsqueeze(1) - normalized_distance, min=0) # Positive if route can accommodate traveling to the node

    # Simple transformations to compute the heuristic scores
    score1 = -normalized_distance + (capacity_penalty * 0.5) + (duration_penalty * 0.5)

    # Apply time windows effects
    current_time = arrival_times + normalized_distance
    time_window_violation = (current_time < time_windows[:, 0].unsqueeze(0)).float() + (current_time > time_windows[:, 1].unsqueeze(0)).float()
    score2 = -time_window_violation * 10

    # Combine scores for a final heuristic score matrix
    heuristic_scores = score1 + score2

    return heuristic_scores