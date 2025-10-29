import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasibility based on load and demand
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    load_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Calculate feasibility based on time windows
    time_feasibility = ((arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times > time_windows[:, 0].unsqueeze(0))).float()

    # Calculate feasibility based on duration limits
    duration_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine feasibility matrices
    feasibility = load_feasibility * time_feasibility * duration_feasibility

    # Calculate heuristic score based on distance and feasibility
    heuristic_scores = -current_distance_matrix * feasibility

    # Introduce randomness through noise
    noise = (torch.rand_like(heuristic_scores) * 0.1)
    
    # Apply noise to scores to enhance exploration and avoid local optima
    heuristic_scores += noise

    # Normalize scores to keep them bounded
    heuristic_scores = heuristic_scores - heuristic_scores.min()
    heuristic_scores = heuristic_scores / heuristic_scores.max()

    return heuristic_scores