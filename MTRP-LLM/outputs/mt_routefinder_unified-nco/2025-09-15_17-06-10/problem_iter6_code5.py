import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)
    rand_weights3 = torch.rand_like(current_distance_matrix)

    normalized_distance = current_distance_matrix / torch.max(current_distance_matrix)

    score1 = torch.sigmoid(torch.tanh(normalized_distance)) * rand_weights1
    score2 = torch.relu(torch.cos(current_distance_matrix)) * rand_weights2
    score3 = torch.tanh(torch.sin(current_distance_matrix)) * torch.exp(rand_weights3)

    heuristic_scores = score1 - score2 + score3

    return heuristic_scores