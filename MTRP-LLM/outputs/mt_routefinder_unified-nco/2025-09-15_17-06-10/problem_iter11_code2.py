import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)

    normalized_distance = current_distance_matrix / torch.max(current_distance_matrix)

    score1 = torch.sigmoid(normalized_distance) * rand_weights1
    score2 = torch.cos(current_distance_matrix) + rand_weights2

    heuristic_scores = score1 - score2

    return heuristic_scores