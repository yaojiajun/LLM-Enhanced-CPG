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
    
    # Initialize the heuristic score matrix with zeros
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Compute potential visitability based on load constraints
    delivery_capacity_reached = (current_load.unsqueeze(1) < delivery_node_demands.unsqueeze(0)).float()
    open_capacity_reached = (current_load_open.unsqueeze(1) < delivery_node_demands_open.unsqueeze(0)).float()
    
    visitable_delivery = 1 - delivery_capacity_reached
    visitable_open = 1 - open_capacity_reached

    # Compute potential visitability based on time windows
    arrival_at_nodes = arrival_times.unsqueeze(1) + current_distance_matrix
    within_time_windows = (arrival_at_nodes >= time_windows[:, 0].unsqueeze(0)).float() * \
                          (arrival_at_nodes <= time_windows[:, 1].unsqueeze(0)).float()
    
    # Compute remaining length constraints
    length_constraints = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine all constraints
    feasible_visits = visitable_delivery * feasible_visits * length_constraints * within_time_windows
    
    # Compute heuristic scores based on distance
    heuristic_scores = feasible_visits * (1 / (current_distance_matrix + 1e-9))  # Avoid division by zero
    
    # Apply randomness to avoid local optima
    noise = torch.rand_like(heuristic_scores) * 0.1  # Small noise
    heuristic_scores += noise

    return heuristic_scores