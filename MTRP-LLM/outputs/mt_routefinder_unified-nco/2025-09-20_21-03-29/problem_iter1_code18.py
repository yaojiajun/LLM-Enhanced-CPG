import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify how distance heuristic, delivery score, and pickup score are computed and integrated into the total score

    # Calculate distance heuristic based on current load and demands
    distance_heuristic = 1 / (current_distance_matrix + 1e-8)

    # Calculate delivery score based on remaining load capacity
    delivery_score = 1 / (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1e-8))

    # Calculate pickup score based on remaining load capacity
    pickup_score = 1 / (pickup_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1e-8))

    # Introduce randomness
    random_noise = torch.rand_like(current_distance_matrix) * 0.1

    # Combine the heuristics with controlled randomness
    total_score = distance_heuristic + delivery_score - pickup_score + random_noise

    return total_score