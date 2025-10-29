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
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate effective distance weights: lighter weights for closer nodes
    effective_distances = current_distance_matrix / (1 + (current_load.unsqueeze(1) / delivery_node_demands.unsqueeze(0)).clamp(0, 1))
    
    # Calculate time window feasibility
    time_window_score = ((arrival_times < time_windows[:, 0].unsqueeze(0)) * (-1.0) + 
                          (arrival_times > time_windows[:, 1].unsqueeze(0)) * (-1.0)).float()
    
    # Calculate load feasibility; positive scores where demands fit
    load_feasibility = ((current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) * 1.0)
    
    # Calculate backhaul feasibility
    backhaul_feasibility = ((current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)) * 1.0)
    
    # Calculate remaining duration feasibility
    duration_feasibility = ((current_length.unsqueeze(1) >= effective_distances) * 1.0)
    
    # Combine scores
    heuristic_scores = (load_feasibility + backhaul_feasibility + duration_feasibility) * effective_distances + time_window_score
    
    # Introduce randomness for exploration:
    random_noise = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += random_noise
    
    return heuristic_scores