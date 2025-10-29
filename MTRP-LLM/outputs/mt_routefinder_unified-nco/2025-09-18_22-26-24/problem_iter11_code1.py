import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Implementing heuristics_v2 based on the provided directions
    noise = torch.rand_like(current_distance_matrix) * 0.1  # Adaptive noise
    day_window_penalty = (time_windows[:, 1] - time_windows[:, 0]) * 0.01  # Penalty for time window violations
    capacity_penalty = delivery_node_demands * 0.02  # Penalty for capacity violations
    
    heuristic_indicators = current_distance_matrix + noise - day_window_penalty.unsqueeze(0) - capacity_penalty.unsqueeze(0)

    return heuristic_indicators