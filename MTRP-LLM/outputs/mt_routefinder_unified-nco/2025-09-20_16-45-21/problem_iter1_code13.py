import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8

    # Calculate delivery feasibility
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Calculate time window feasibility
    time_feasibility = ((arrival_times.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0)) & 
                        (arrival_times.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0))).float()

    # Calculate route length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Calculate overall feasibility score
    feasibility_score = delivery_feasibility * time_feasibility * length_feasibility

    # Compute heuristic scores based on distance and feasibility
    base_scores = (1 / (current_distance_matrix + epsilon)) * feasibility_score

    # Incorporate randomness to avoid local optima
    randomness = torch.normal(mean=0.0, std=0.1, size=base_scores.shape).to(base_scores.device)
    
    heuristic_scores = base_scores + randomness

    # Clamp scores to ensure all values are finite
    heuristic_scores = torch.clamp(heuristic_scores, min=float('-inf'), max=float('inf'))
    heuristic_scores[~torch.isfinite(heuristic_scores)] = float('-inf')

    return heuristic_scores