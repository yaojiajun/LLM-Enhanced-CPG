import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    
    # Compute remaining capacity for deliveries
    remaining_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    feasible_deliveries = (remaining_capacity >= 0).float()
    
    # Compute time window feasibility
    earliest_arrival = arrival_times + current_distance_matrix
    time_window_feasibility = ((earliest_arrival >= time_windows[:, 0].unsqueeze(0)) & 
                                (earliest_arrival <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Compute remaining length budget allowing for the current trajectory
    remaining_length = current_length.unsqueeze(1) - current_distance_matrix
    
    # Ensure valid remaining length
    valid_length = (remaining_length >= 0).float()
    
    # Score calculation: Combine features
    scores = (feasible_deliveries * time_window_feasibility * valid_length) / (current_distance_matrix + epsilon)
    
    # Incorporate randomness to avoid premature convergence
    randomness = torch.rand_like(scores) * 0.1  # Control randomness impact
    scores += randomness

    # Ensure scores contain only finite values
    scores = torch.clamp(scores, min=-1e8, max=1e8)

    return scores