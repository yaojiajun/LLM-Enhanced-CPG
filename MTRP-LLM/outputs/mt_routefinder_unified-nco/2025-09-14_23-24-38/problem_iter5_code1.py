import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Improved heuristics implementation incorporating the given directions
    constraint_severity = torch.sum(current_load) + torch.sum(current_length)  # Example constraint severity calculation
    randomness_scale = torch.rand_like(current_distance_matrix) * constraint_severity  # Adaptive scaling for randomness

    historical_performance = torch.mean(current_distance_matrix)  # Historical performance data example
    dynamic_scores = torch.exp(current_distance_matrix / historical_performance)  # Dynamic scoring mechanism

    deterministic_scores = 1 / torch.arange(1, current_distance_matrix.size(1) + 1, dtype=torch.float)  # Deterministic score example

    heuristic_scores = dynamic_scores + randomness_scale - deterministic_scores  # Combined heuristic scores

    return heuristic_scores