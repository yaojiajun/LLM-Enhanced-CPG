import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Introduce adaptive scaling for exploration and exploitation
    exploration_factor = torch.rand_like(current_distance_matrix) * 0.05
    exploitation_factor = torch.randn_like(current_distance_matrix) * 0.1

    # Introduce penalty adjustments for constraint violations
    penalty_factor = torch.zeros_like(current_distance_matrix)
    penalty_factor[(current_load < 0) | (current_load_open < 0) | (current_length < 0)] = 1.0

    # Combine factors to compute heuristic scores
    heuristic_scores = current_distance_matrix * (1 + exploration_factor) + exploitation_factor - penalty_factor

    # Introduce learned weights for exploration-exploitation balance
    weighted_scores = heuristic_scores * 0.8 + torch.randn_like(heuristic_scores) * 0.2

    return weighted_scores