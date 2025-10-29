import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    epsilon = 1e-8

    available_capacity = current_load.unsqueeze(-1) - delivery_node_demands.unsqueeze(0)
    open_route_capacity = current_load_open.unsqueeze(-1) - delivery_node_demands_open.unsqueeze(0)

    feasible_delivery_mask = (available_capacity >= 0).float()
    feasible_open_mask = (open_route_capacity >= 0).float()

    # Time window validities
    earliest_departure = arrival_times + current_distance_matrix
    time_window_mask = ((earliest_departure >= time_windows[:, 0].unsqueeze(0)) & 
                        (earliest_departure <= time_windows[:, 1].unsqueeze(0))).float()

    # Duration limitations
    duration_limits = current_length.unsqueeze(-1) - current_distance_matrix
    duration_mask = (duration_limits >= 0).float()

    # Heuristic score: favor feasible routes, scale by travel distances, inverse of available time, remaining load & distance for balancing
    heuristic_scores = feasible_delivery_mask * time_window_mask * duration_mask / (current_distance_matrix + epsilon)

    # Incorporating randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.01
    heuristic_scores += randomness

    # Clamping to make sure all values are finite
    heuristic_scores = torch.clamp(heuristic_scores, min=-1e5, max=1e5)

    return heuristic_scores