import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasibility factors
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    load_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Calculate time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasibility = ((current_time >= time_windows[:, 0].unsqueeze(0)) & 
                                (current_time <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Calculate route length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Calculate desirability scores based on feasibility
    desirability_scores = load_feasibility * load_feasibility_open * time_window_feasibility * length_feasibility

    # Incorporate distance into heuristic scores (lower distances are better)
    distance_scores = 1 / (current_distance_matrix + 1e-5)  # Avoid division by zero, ensuring scores are positive
    
    # Combine desirability and distance scores
    heuristic_scores = desirability_scores * distance_scores

    # Introduce randomness to avoid local optima
    randomness = 0.1 * torch.rand_like(heuristic_scores)
    heuristic_scores += randomness

    return heuristic_scores