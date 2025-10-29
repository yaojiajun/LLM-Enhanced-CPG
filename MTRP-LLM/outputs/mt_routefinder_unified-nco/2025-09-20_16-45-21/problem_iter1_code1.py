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
    
    # Initialize the heuristic score matrix
    num_vehicles, num_nodes = current_distance_matrix.shape
    heuristic_scores = torch.zeros(num_vehicles, num_nodes, device=current_distance_matrix.device)
    
    # Calculate the permissible for delivery based on current load and delivery demands
    deliverable_mask = (current_load.unsqueeze(1) >= (delivery_node_demands.unsqueeze(0) + epsilon))
    # Randomness to avoid local optima
    randomness = torch.rand_like(current_distance_matrix) * 0.1
  
    # Score for capacity feasibility
    capacity_scores = deliverable_mask.float() * (1 - current_distance_matrix / (current_length.unsqueeze(1) + epsilon))
    
    # Calculate time window feasibility
    time_window_mask = ((arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) &
                        (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)))
    
    # Score for time window compliance
    time_window_scores = time_window_mask.float()
    
    # Calculate backhaul feasibility (assuming backhaul nodes have negative demands)
    backhaul_mask = (pickup_node_demands.unsqueeze(0) <= current_load.unsqueeze(1))
    backhaul_scores = backhaul_mask.float()
    
    # Combine scores: maximize delivery feasible routes and minimize distance
    heuristic_scores = (capacity_scores + time_window_scores + backhaul_scores) * (1 - current_distance_matrix) + randomness
    
    # Mask invalid scores
    heuristic_scores[~(deliverable_mask & time_window_mask & backhaul_mask)] = float('-inf')
    
    # Clamp to finite values
    heuristic_scores = torch.clamp(heuristic_scores, min=float('-1e10'), max=float('1e10'))
    
    return heuristic_scores