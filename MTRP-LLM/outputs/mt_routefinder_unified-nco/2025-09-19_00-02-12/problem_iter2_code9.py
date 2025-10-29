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
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Check capacity constraints for deliveries
    capacity_constraints = (current_load[:, None] >= delivery_node_demands[None, :]) & (current_load[:, None] >= 0)
    
    # Check capacity constraints for pickups
    capacity_constraints_open = (current_load_open[:, None] >= pickup_node_demands[None, :]) & (current_load_open[:, None] >= 0)

    # Check time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_constraints = (current_time >= time_windows[:, 0][None, :]) & (current_time <= time_windows[:, 1][None, :])

    # Calculate the intersection of all constraints (Capacity for delivery & time window feasibility)
    valid_delivery = capacity_constraints & time_window_constraints
    valid_pickup = capacity_constraints_open & time_window_constraints

    # Score positive for valid edges (make them more promising)
    heuristic_scores[valid_delivery] += 1.0  # Increment for feasible deliveries
    heuristic_scores[valid_pickup] += 1.0     # Increment for feasible pickups

    # Add a component for travel distance (prefer shorter distances)
    heuristic_scores -= current_distance_matrix  

    # Add randomness to enhance exploration
    randomness_factor = torch.rand_like(heuristic_scores) * 0.05
    heuristic_scores += randomness_factor
    
    return heuristic_scores