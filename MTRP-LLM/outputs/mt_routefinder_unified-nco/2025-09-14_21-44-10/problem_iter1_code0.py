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
    
    # Initialize the heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate available capacity for deliveries and pickups
    available_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    available_capacity_open = current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)

    # Mask infeasible routes based on vehicle capacity
    capacity_mask = (available_capacity >= 0) & (available_capacity_open >= 0)
    
    # Calculate time window feasibility
    arrival_time_mask = (arrival_times <= time_windows[:, 1].unsqueeze(0)) & (arrival_times >= time_windows[:, 0].unsqueeze(0))
    
    # Duration feasibility
    duration_mask = (current_length.unsqueeze(1) >= current_distance_matrix)
    
    # Combine all masks
    feasibility_mask = capacity_mask & arrival_time_mask & duration_mask
    
    # Calculate the heuristic scores
    scores = torch.where(feasibility_mask, 
                         -current_distance_matrix + 10 * (1 - feasibility_mask.float()), 
                         torch.tensor(float('-inf')).expand_as(current_distance_matrix))

    # Enhance randomness to avoid local optima
    randomness = torch.rand_like(scores) * 0.1
    heuristic_scores = scores + randomness

    # Ensure that we do not recommend visiting the depot node (node 0)
    heuristic_scores[:, 0] = float('-inf')

    return heuristic_scores