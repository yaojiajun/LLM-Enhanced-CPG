import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Random weights for enhanced randomness
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)

    # Normalized distance matrix
    max_distance = torch.max(current_distance_matrix, dim=1, keepdim=True).values
    normalized_distance = current_distance_matrix / max_distance

    # Heuristic components
    score1 = torch.tanh(normalized_distance) * rand_weights1

    # Incorporating delivery demands and time window penalties
    demand_penalty = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * 2.0
    time_window_penalty = (arrival_times > time_windows[:, 1].unsqueeze(0)).float() * 3.0

    score2 = (torch.exp(current_distance_matrix) + demand_penalty + time_window_penalty) * rand_weights2

    # Adding random exploration
    random_exploration = (torch.rand_like(current_distance_matrix) - 0.5) * 0.5

    # Final heuristic score calculation
    heuristic_scores = score1 - score2 + random_exploration

    return heuristic_scores