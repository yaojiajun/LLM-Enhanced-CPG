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
    
    # Calculate initial scores based on distance
    base_scores = -current_distance_matrix
    
    # Apply delivery demand constraints
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Calculate waiting times based on arrival times and time windows
    waiting_times = torch.clamp(time_windows[:, 0] - arrival_times, min=0) + torch.clamp(arrival_times - time_windows[:, 1], min=0)
    
    # Generate a randomness factor to introduce exploration
    random_factor = torch.rand_like(current_distance_matrix) * torch.randn_like(current_distance_matrix) * 0.05
    
    # Calculate pickup feasibility
    pickup_feasibility = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= current_load_open.unsqueeze(1)).float()

    # Create a combined score
    score_matrix = base_scores + (delivery_feasibility * pickup_feasibility) - waiting_times + random_factor
    
    # Introduce adaptive penalty based on feasibility
    adaptive_penalty = (1 - delivery_feasibility) * 0.2 + (1 - pickup_feasibility) * 0.2
    score_matrix -= adaptive_penalty
    
    return score_matrix