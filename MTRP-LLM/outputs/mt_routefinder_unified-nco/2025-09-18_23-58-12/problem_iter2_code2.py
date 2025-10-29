import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Calculate heuristic score matrix based on a combination of current distance matrix and randomness
    random_scores = 0.1 * torch.randn_like(current_distance_matrix)
    heuristic_scores = current_distance_matrix + random_scores

    return heuristic_scores