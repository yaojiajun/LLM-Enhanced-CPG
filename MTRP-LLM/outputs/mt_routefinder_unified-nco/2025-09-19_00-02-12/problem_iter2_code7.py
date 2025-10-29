import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic scores with random values for exploration
    heuristic_scores = torch.randn_like(current_distance_matrix)

    # Calculate remaining capacity related heuristic
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()
    
    # Calculate time window feasibility
    current_time = arrival_times
    in_time_window = ((current_time <= time_windows[:, 1].unsqueeze(0)) & (current_time >= time_windows[:, 0].unsqueeze(0))).float()

    # Penalize edges that are not feasible due to load and time constraints
    heuristic_scores *= feasible_delivery * feasible_pickup * in_time_window

    # Adjust scores based on distance - closer nodes get higher positive scores
    max_distance = torch.max(current_distance_matrix)
    distance_penalty = 1 - (current_distance_matrix / max_distance)
    
    heuristic_scores += distance_penalty

    # Add randomness to avoid local optima while ensuring positive scores
    random_factor = 0.1 * torch.randn_like(heuristic_scores)
    heuristic_scores += random_factor

    # Ensure scores are non-negative for promising edges
    heuristic_scores = torch.clamp(heuristic_scores, min=0)

    return heuristic_scores