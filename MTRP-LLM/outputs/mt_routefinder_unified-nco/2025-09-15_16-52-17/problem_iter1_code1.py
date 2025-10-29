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
    
    # Heuristic score initialization
    score_matrix = torch.zeros_like(current_distance_matrix)

    # Identify feasible nodes based on capacity constraints (delivery and pickup)
    feasible_deliveries = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    feasible_pickups = (current_load.unsqueeze(1) >= pickup_node_demands.unsqueeze(0))
    
    # Time feasibility
    current_time = arrival_times + current_distance_matrix
    time_feasible = (current_time <= time_windows[:, 1].unsqueeze(0)) & (current_time >= time_windows[:, 0].unsqueeze(0))
    
    # Combine constraints
    feasible_nodes = feasible_deliveries & feasible_pickups & time_feasible
    
    # Calculate penalties for infeasible nodes
    penalties = -1e10 * (~feasible_nodes).float()
    
    # Additional scoring based on distance heuristics, introduce randomness for exploration
    distance_scores = -current_distance_matrix + penalties
    randomness = torch.rand_like(distance_scores) * 0.1
    
    # Final score matrix with added randomness
    score_matrix = distance_scores + randomness

    # Ensure the infeasible nodes maintain a low score
    score_matrix[~feasible_nodes] = penalized_value = -1e10

    return score_matrix