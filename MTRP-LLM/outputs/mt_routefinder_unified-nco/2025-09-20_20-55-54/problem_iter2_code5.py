import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize the distance matrix
    norm_distance_matrix = current_distance_matrix / (current_distance_matrix.max() + 1e-8)

    # Introduce adaptive noise levels based on exploration-exploitation trade-off
    exploration_rate = torch.rand(1) * 0.2  # Random exploration rate between 0 and 0.2
    noise = torch.rand_like(norm_distance_matrix) * exploration_rate
    norm_distance_matrix += noise

    # Apply heuristic functions to generate scores
    heuristic_scores = norm_distance_matrix * 0.8  # Example heuristic score calculation

    return heuristic_scores