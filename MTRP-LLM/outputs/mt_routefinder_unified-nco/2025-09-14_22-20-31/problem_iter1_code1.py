import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize scores with negative distances
    scores = -current_distance_matrix
    
    # Check delivery demand feasibility for current loads
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    scores += feasible_delivery * 100.0  # Boost feasible routes
    
    # Check open load capacity feasibility
    feasible_delivery_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    scores += feasible_delivery_open * 100.0  # Boost feasible open routes

    # Check time window feasibility
    arrival_time_adjusted = arrival_times.unsqueeze(1) + current_distance_matrix
    within_time_windows = (arrival_time_adjusted >= time_windows[:, 0].unsqueeze(0)) & (arrival_time_adjusted <= time_windows[:, 1].unsqueeze(0))
    scores += within_time_windows.float() * 50.0  # Boost routes within time windows

    # Penalize for waiting times based on time windows
    waiting_time_penalty = torch.clamp(time_windows[:, 0].unsqueeze(0) - arrival_times.unsqueeze(1), min=0)
    scores -= waiting_time_penalty * 10.0  # Apply waiting time penalties

    # Check length constraints
    feasible_length = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    scores += feasible_length * 20.0  # Boost feasible lengths

    # Incorporate randomness to avoid local optima
    random_noise = torch.rand(scores.shape, device=scores.device) * 5.0  # Add small random noise
    scores += random_noise

    return scores