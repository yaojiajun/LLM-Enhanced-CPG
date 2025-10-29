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

    # Distance heuristics
    distance_score = 1 / (current_distance_matrix + epsilon)

    # Load and demand heuristics
    demand_capacity_score = (current_load.unsqueeze(1) - delivery_node_demands) / (current_load.unsqueeze(1) + epsilon)
    demand_capacity_open_score = (current_load_open.unsqueeze(1) - delivery_node_demands_open) / (current_load_open.unsqueeze(1) + epsilon)
    
    # Time window checks
    time_window_score = torch.where((arrival_times <= time_windows[:, 1].unsqueeze(0)) & 
                                     (arrival_times >= time_windows[:, 0].unsqueeze(0)), 
                                     torch.ones_like(arrival_times), 
                                     torch.zeros_like(arrival_times))

    # Route length constraints
    route_length_score = current_length.unsqueeze(1) / (current_distance_matrix + epsilon)

    # Pickup node demands handling
    pickup_score = 1 / (pickup_node_demands + epsilon)

    # Combining scores while applying a mask for infeasibility
    score_matrix = (distance_score + 
                    demand_capacity_score + 
                    demand_capacity_open_score + 
                    time_window_score + 
                    route_length_score + 
                    pickup_score)

    # Controlled randomness to avoid local optima
    randomness = torch.rand(score_matrix.shape) * 0.1
    score_matrix += randomness

    # Final masking for non-finite values
    score_matrix = torch.where(torch.isfinite(score_matrix), score_matrix, torch.tensor(-float('inf')).to(score_matrix.device))

    return score_matrix