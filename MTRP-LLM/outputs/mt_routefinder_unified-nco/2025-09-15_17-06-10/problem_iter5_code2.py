import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Compute normalized distance
    normalized_distance = current_distance_matrix / (current_distance_matrix.max(dim=-1, keepdim=True).values + 1e-5)

    # Evaluate delivery feasibility
    delivery_feasibility = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility computation
    time_feasibility = (arrival_times < time_windows[:, 1].unsqueeze(0)).float() * (arrival_times >= time_windows[:, 0].unsqueeze(0)).float()

    # Compute a randomness component
    rand_weights = torch.rand_like(current_distance_matrix) * 0.5 + 0.5

    # Compute heuristic scores with multiple indicators
    score_distance = torch.tanh(normalized_distance) * rand_weights
    score_delivery = delivery_feasibility * rand_weights * 0.8
    score_time = time_feasibility * (1 - rand_weights) * 0.8
    score_capacity = (current_length.unsqueeze(-1) >= current_distance_matrix) * (1 - rand_weights) * 0.6
    
    # Combine scores
    heuristic_scores = score_distance + score_delivery + score_time + score_capacity
    
    # Normalize scores
    heuristic_scores = (heuristic_scores - heuristic_scores.min(dim=-1, keepdim=True).values) / (heuristic_scores.max(dim=-1, keepdim=True).values - heuristic_scores.min(dim=-1, keepdim=True).values + 1e-5)
    
    # Apply randomness to avoid local optima
    heuristic_scores += (torch.rand_like(heuristic_scores) * 0.1)  # Introduce slight noise
    
    return heuristic_scores