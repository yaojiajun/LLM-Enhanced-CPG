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

    # Calculate feasibility masks based on capacity and time window constraints
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0))
    time_feasibility = (arrival_times <= time_windows[:, 1].unsqueeze(0)) & (arrival_times >= time_windows[:, 0].unsqueeze(0))
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)

    feasible_edges = load_feasibility & time_feasibility & length_feasibility

    # Calculate base heuristic scores based on distance and feasibility
    base_scores = torch.where(feasible_edges, -current_distance_matrix, torch.full_like(current_distance_matrix, float('inf')))
    
    # Introduce adaptive penalty mechanisms based on solution feasibility
    penalty_scores = torch.zeros_like(base_scores)
    penalty_mask = ~feasible_edges
    penalty_scores[penalty_mask] = (current_distance_matrix[penalty_mask] * 0.2)  # Example penalty

    # Random scores to enhance exploration
    random_scores = torch.rand_like(current_distance_matrix) * 0.1

    # Combine scores with penalties
    heuristic_scores = base_scores + penalty_scores + random_scores

    # Return scores, ensuring to keep them within a defined range
    return heuristic_scores.clamp(min=-1.0, max=0.0)