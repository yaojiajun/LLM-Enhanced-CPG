import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Constants for penalties and rewards
    capacity_penalty = 1000
    time_window_penalty = 500
    length_penalty = 200
    random_factor = torch.rand(current_distance_matrix.shape)

    # Calculate feasibility conditions
    sufficient_capacity_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    sufficient_capacity_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()
    
    current_time = arrival_times.unsqueeze(1)
    earliest_time = time_windows[:, 0].unsqueeze(0)
    latest_time = time_windows[:, 1].unsqueeze(0)

    within_time_windows = (current_time >= earliest_time) & (current_time <= latest_time)
    
    # Compute scores based on distance and penalties
    distance_scores = -current_distance_matrix
    capacity_scores = -sufficient_capacity_delivery * capacity_penalty
    time_window_scores = -((1 - within_time_windows.float()) * time_window_penalty)
    length_scores = -((current_length.unsqueeze(1) < current_distance_matrix).float() * length_penalty)

    # Combining scores
    heuristic_scores = distance_scores + capacity_scores + time_window_scores + length_scores + random_factor
    
    return heuristic_scores