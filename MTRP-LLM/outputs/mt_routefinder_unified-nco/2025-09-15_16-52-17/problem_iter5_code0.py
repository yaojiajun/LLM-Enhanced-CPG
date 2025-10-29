import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    exploration_factor = 0.1
    random_scores = torch.rand_like(current_distance_matrix)
    penalty_scores = torch.rand_like(current_distance_matrix) * exploration_factor
    exploration_scores = torch.rand_like(current_distance_matrix) * exploration_factor
    heuristic_scores = random_scores - penalty_scores + exploration_scores

    return heuristic_scores