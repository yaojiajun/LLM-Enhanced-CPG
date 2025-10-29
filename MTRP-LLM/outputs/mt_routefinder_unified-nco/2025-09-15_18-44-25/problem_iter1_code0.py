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

    # Compute delivery feasibility based on current load and demands
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Compute time window feasibility
    time_window_feasibility = ((arrival_times <= time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times >= time_windows[:, 0].unsqueeze(0))).float()

    # Compute remaining length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Calculate base distance score (inverted distances to prioritize shorter distances)
    distance_scores = 1.0 / (current_distance_matrix + 1e-6)  # Adding a small number to prevent division by zero

    # Calculate total feasibility score by combining all feasibility indicators
    feasibility_scores = delivery_feasibility * delivery_feasibility_open * time_window_feasibility * length_feasibility

    # Combine feasibility scores with distance scores to create the final heuristic score
    score_matrix = feasibility_scores * distance_scores

    # Introduce randomness to avoid local optima (adding noise)
    noise = torch.randn_like(score_matrix) * 0.1
    score_matrix += noise

    # Normalize scores to keep them within bounds
    score_matrix = torch.clamp(score_matrix, min=-1, max=1)

    return score_matrix