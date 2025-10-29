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
    score_matrix = torch.zeros_like(current_distance_matrix)
    
    # Calculate distance-based scores (lower distance is better)
    distance_scores = -current_distance_matrix
    
    # Capacity constraints handling
    capacity_constraints_delivery = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    capacity_constraints_pickup = (current_load_open.unsqueeze(-1) >= pickup_node_demands.unsqueeze(0)).float()
    
    # Apply penalties for violating capacity constraints
    score_matrix += distance_scores * capacity_constraints_delivery
    score_matrix += distance_scores * capacity_constraints_pickup
    
    # Time windows handling
    in_time_window = ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float() * \
                     ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)).float()
    
    # Apply penalties for violating time window constraints
    score_matrix += distance_scores * in_time_window
    
    # Duration limits handling
    duration_conditions = (current_length.unsqueeze(-1) >= current_distance_matrix).float()
    
    # Apply penalties for violating duration limits
    score_matrix += distance_scores * duration_conditions
    
    # Introduce randomness to avoid local optima
    randomness = torch.rand_like(score_matrix) * 0.1  # Small random perturbation
    score_matrix += randomness
    
    return score_matrix