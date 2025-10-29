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
    
    # Calculate potential utility based on demands and remaining capacity for deliveries
    delivery_utility = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float() * \
                       (1 - (current_distance_matrix / current_distance_matrix.max()))
                       
    # Calculate potential utility for pickups
    pickup_utility = (current_load_open.unsqueeze(-1) >= pickup_node_demands.unsqueeze(0)).float() * \
                     (1 - (current_distance_matrix / current_distance_matrix.max()))
    
    # Time windows violation penalties
    time_window_penalty = ((arrival_times.unsqueeze(-1) < time_windows[:, 0]) | 
                           (arrival_times.unsqueeze(-1) > time_windows[:, 1])).float() * 1000

    # Route length consideration
    length_penalty = (current_length.unsqueeze(-1) < current_distance_matrix).float() * 1000
    
    # Combined heuristic score including utilities and penalties
    heuristic_scores = delivery_utility + pickup_utility - time_window_penalty - length_penalty
    
    # Introduce randomness to avoid local optima
    random_factor = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += random_factor
    
    return heuristic_scores