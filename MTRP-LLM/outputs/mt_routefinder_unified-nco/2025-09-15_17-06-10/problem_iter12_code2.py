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
    
    # Introduce randomness to enhance exploration
    rand_weights = torch.rand_like(current_distance_matrix) * 0.1

    # Normalize the distance matrix
    max_distance = torch.max(current_distance_matrix, dim=1, keepdim=True).values
    normalized_distance = current_distance_matrix / (max_distance + 1e-6)  # Adding small epsilon to avoid division by zero

    # Calculate the service feasibility based on demand and capacity
    feasibility = (current_load.unsqueeze(1) >= delivery_node_demands) & (current_load_open.unsqueeze(1) >= pickup_node_demands)
    
    # Time window feasibility
    time_feasibility = (arrival_times + current_distance_matrix <= time_windows[:, 1]) & \
                       (arrival_times + current_distance_matrix >= time_windows[:, 0])
    
    # Combining feasibility checks into a score
    feasibility_score = feasibility.float() * time_feasibility.float()

    # Create score based on normalized distances and feasibility
    score = (1 - normalized_distance) * feasibility_score + rand_weights

    # Encourage exploration by introducing a penalty for visited nodes (considering they cannot be re-visited)
    visited_penalty = (current_length.unsqueeze(1) < current_distance_matrix).float() * -1.0
    score += visited_penalty

    return score