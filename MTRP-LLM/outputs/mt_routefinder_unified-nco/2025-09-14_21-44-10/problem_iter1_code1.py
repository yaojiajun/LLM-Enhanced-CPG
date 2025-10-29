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
    
    # Compute feasible visits based on capacity and time windows
    capacity_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * \
                           (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility based on arrival times
    time_window_feasibility = ((arrival_times.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 0].unsqueeze(0))).float()
    
    # Length feasibility based on current route length
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine feasibility criteria
    feasibility_mask = capacity_feasibility * time_window_feasibility * length_feasibility
    
    # Calculate heuristic scores based on distance, including random noise
    distance_scores = -current_distance_matrix * feasibility_mask # Negative distance for scoring
    random_noise = torch.rand_like(distance_scores) * 0.1  # Small random perturbation for exploration
    heuristic_scores = distance_scores + random_noise
    
    # Apply feasibility mask to enforce invalid options
    heuristic_scores = heuristic_scores * feasibility_mask
    
    return heuristic_scores