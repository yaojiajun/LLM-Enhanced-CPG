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
    
    # Initialize heuristic score matrix filled with zeros
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Compute delivery feasibility based on remaining load and demands
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_window_feasibility = ((arrival_times.unsqueeze(2) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0).unsqueeze(1)) & 
                                (arrival_times.unsqueeze(2) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0).unsqueeze(1))).float()

    # Combine feasibility matrices
    feasibility_matrix = delivery_feasibility * time_window_feasibility * delivery_feasibility_open

    # Calculate a base heuristic score from distance
    base_scores = -current_distance_matrix * feasibility_matrix  # Inverse because lower distances are better
    
    # Incorporate randomness for exploration: adding a small random value
    random_noise = torch.rand_like(base_scores) * 0.01
    enhanced_scores = base_scores + random_noise
    
    # Penalizing exceeding current length budget and capacity
    length_penalty = (current_length.unsqueeze(1) < current_distance_matrix).float() * 1e4  # large penalty
    capacity_penalty = (current_load.unsqueeze(1) < delivery_node_demands.unsqueeze(0)).float() * 1e4  # large penalty
    
    # Adjust final scores with penalties
    heuristic_scores = enhanced_scores - length_penalty - capacity_penalty
    
    return heuristic_scores