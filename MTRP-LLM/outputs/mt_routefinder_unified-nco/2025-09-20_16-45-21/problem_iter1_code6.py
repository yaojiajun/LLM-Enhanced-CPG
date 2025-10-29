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

    # Compute feasible delivery indicators
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Compute feasible open route indicators
    feasible_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_now = arrival_times.unsqueeze(1) + current_distance_matrix
    time_window_feasibility = ((time_now >= time_windows[:, 0].unsqueeze(0)) & 
                                (time_now <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Effective cost calculation for edges
    costs = current_distance_matrix * (feasible_delivery * time_window_feasibility)
    
    # Include pickup node demands as a penalty
    pickup_penalty = pickup_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + epsilon)
    
    # Calculate heuristic scores
    heuristic_scores = (1 / (costs + epsilon)) - pickup_penalty
    
    # Randomness for exploration
    randomness = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += randomness
    
    # Clamp scores to finite bounds
    heuristic_scores = torch.clamp(heuristic_scores, min=float('-inf'), max=float('inf'))

    return heuristic_scores