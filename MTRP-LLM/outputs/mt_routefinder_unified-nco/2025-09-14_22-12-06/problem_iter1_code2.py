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
    
    N_plus_1 = current_distance_matrix.size(1)
    pomo_size = current_distance_matrix.size(0)
    
    # Compute basic feasibility
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_feasibility = ((current_time >= time_windows[:, 0].unsqueeze(0)) & 
                        (current_time <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Calculate heuristic scores
    feasibility_scores = load_feasibility * length_feasibility * time_feasibility
    
    # Give negative scores for infeasible options
    infeasibility_penalty = (1 - feasibility_scores) * 1000  # Large penalty for infeasibility
    heuristic_scores = feasibility_scores * (1 / (current_distance_matrix + 1e-6)) - infeasibility_penalty
    
    # Introduce randomness to avoid local optima
    random_factor = torch.rand(pomo_size, N_plus_1) * 0.1  # Adding small random perturbation
    heuristic_scores += random_factor
    
    return heuristic_scores