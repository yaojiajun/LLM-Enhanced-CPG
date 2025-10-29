import torch
import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize the distance matrix
    norm_distance_matrix = current_distance_matrix / (current_distance_matrix.max() + 1e-8)

    # Randomly add noise for exploration
    noise = torch.rand_like(norm_distance_matrix) * 0.1 # 10% noise
    norm_distance_matrix = norm_distance_matrix + noise

    # Apply heuristic functions to generate scores
    heuristic_score = norm_distance_matrix * 0.8  # Example heuristic score calculation

    return heuristic_score