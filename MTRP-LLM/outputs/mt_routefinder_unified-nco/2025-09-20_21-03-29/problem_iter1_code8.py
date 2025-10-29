import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    distance_heuristic = 1 / (current_distance_matrix + 1e-8)  # Adjusted distance heuristic calculation
    delivery_score = delivery_node_demands / (current_load + 1e-8)  # Adjusted delivery score calculation
    pickup_score = pickup_node_demands / (current_load + 1e-8)  # Adjusted pickup score calculation

    total_score = distance_heuristic + delivery_score - pickup_score
    return total_score