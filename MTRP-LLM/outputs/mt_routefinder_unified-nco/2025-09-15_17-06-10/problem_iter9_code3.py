import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalization Factor
    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / (max_distance + 1e-8)  # Avoid division by zero

    # Heuristic Components
    # Component 1: Delivery feasibility considering demand and load
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()

    # Component 2: Time window feasibility
    current_time = arrival_times[:, 0].unsqueeze(1)  # Assume starting from node 0
    time_window_feasibility = ((current_time >= time_windows[:, 0].unsqueeze(0)) & 
                                (current_time <= time_windows[:, 1].unsqueeze(0))).float()

    # Component 3: Remaining length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Randomness components
    rand_weights = torch.rand_like(current_distance_matrix)

    # Scoring
    score_delivery = delivery_feasibility * (1 - normalized_distance) * rand_weights
    score_time_window = time_window_feasibility * (1 - normalized_distance)
    score_length = length_feasibility * (1 - normalized_distance) * (1 + rand_weights)

    # Combine Scores
    heuristic_scores = score_delivery + score_time_window + score_length

    # Introduce randomness to avoid local optima
    heuristic_scores += (torch.rand_like(heuristic_scores) * 0.1)  # Small random perturbation

    return heuristic_scores