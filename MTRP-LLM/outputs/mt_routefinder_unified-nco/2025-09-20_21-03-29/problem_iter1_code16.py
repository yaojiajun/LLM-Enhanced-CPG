import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    distance_heuristic = torch.reciprocal(current_distance_matrix + 1e-8)

    # Calculate delivery score
    remaining_capacity = current_load.unsqueeze(1) - delivery_node_demands
    delivery_score = torch.where(remaining_capacity >= 0, 1 / (remaining_capacity + 1e-8), -1e8)

    # Calculate pickup score
    pickup_score = torch.reciprocal(pickup_node_demands + 1e-8)

    # Combine heuristic indicators
    total_score = distance_heuristic + delivery_score + pickup_score
    return total_score