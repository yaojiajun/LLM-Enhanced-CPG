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
    
    pomo_size, num_nodes = current_distance_matrix.shape
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)
    
    # Evaluating feasibility based on delivery demands and current load
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_delivery_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_window_open = (arrival_times + current_distance_matrix).clamp(min=0) >= time_windows[:, 0].unsqueeze(0)
    time_window_close = (arrival_times + current_distance_matrix).clamp(min=0) <= time_windows[:, 1].unsqueeze(0)
    feasible_time_window = (time_window_open & time_window_close).float()
    
    # Route duration feasibility
    feasible_duration = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combined feasibility
    combined_feasibility = feasible_delivery * feasible_delivery_open * feasible_time_window * feasible_duration
    
    # Calculate heuristic scores
    heuristic_scores += combined_feasibility * (-current_distance_matrix)
    
    # Introduce randomness to avoid convergence to local optima
    randomness_factor = torch.rand_like(heuristic_scores) * 0.01  # Small random noise
    heuristic_scores += randomness_factor
    
    # Apply a scaling factor to prioritize certain routes
    route_priority = (combined_feasibility.sum(1, keepdim=True) > 0).float()
    heuristic_scores *= route_priority
    
    return heuristic_scores