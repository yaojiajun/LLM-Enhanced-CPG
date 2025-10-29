import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the distance heuristic calculation
    distance_heuristic = 1.0 / (current_distance_matrix + 1e-8)

    # Modify the delivery score calculation
    delivery_score = 1.0 / (delivery_node_demands + 1e-8)

    # Modify the pickup score calculation
    pickup_score = 1.0 / (pickup_node_demands + 1e-8)

    # Integrate the modified heuristics into the total score
    total_score = distance_heuristic + delivery_score - pickup_score

    return total_score