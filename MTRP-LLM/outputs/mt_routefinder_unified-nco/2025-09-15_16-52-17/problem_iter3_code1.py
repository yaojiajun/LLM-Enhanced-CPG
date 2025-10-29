import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Compute basic heuristic scores
    distance_scores = -current_distance_matrix  # Negative distances for shorter routes
    load_scores = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()  # Capacity feasible
    time_scores = ((arrival_times <= time_windows[:, 1].unsqueeze(0)) & (arrival_times >= time_windows[:, 0].unsqueeze(0))).float()  # Time window feasible
    pickup_scores = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= current_load_open.unsqueeze(1)).float()  # Capacity for pickups

    # Combine scores with weight adjustments
    heuristic_scores = (distance_scores * load_scores * time_scores * pickup_scores)

    # Adaptive penalty mechanism
    infeasibility_penalty = 1 - load_scores * time_scores * pickup_scores  # Penalize infeasible options
    adaptive_penalty = torch.mean(heuristic_scores) * infeasibility_penalty

    # Randomness for exploration
    random_scores = torch.randn_like(heuristic_scores) * 0.05  # Small Gaussian noise
    heuristic_scores += adaptive_penalty + random_scores

    # Ensure scores are non-negative, favoring feasible options
    heuristic_scores = torch.clamp(heuristic_scores, min=0.0)

    return heuristic_scores