import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Introduce noise with adaptive scaling based on the current distance matrix values
    noise_scale = 0.05 * torch.mean(current_distance_matrix)
    noise = torch.randn_like(current_distance_matrix) * noise_scale

    # Calculate heuristic score matrix based on distance matrix and noise
    heuristic_scores = current_distance_matrix + noise

    return heuristic_scores