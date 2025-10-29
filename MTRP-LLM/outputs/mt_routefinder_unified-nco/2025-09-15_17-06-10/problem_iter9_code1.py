import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalize the distance matrix for scoring
    max_distances = torch.max(current_distance_matrix, dim=-1, keepdim=True).values
    normalized_distance = current_distance_matrix / max_distances

    # Calculate time window penalties
    soft_time_penalty = torch.where(arrival_times < time_windows[:, 0], time_windows[:, 0] - arrival_times, torch.zeros_like(arrival_times))
    hard_time_penalty = torch.where(arrival_times > time_windows[:, 1], arrival_times - time_windows[:, 1], torch.zeros_like(arrival_times))

    # Calculate load penalties for delivery and pickup
    delivery_penalty = torch.where(current_load[:, None] < delivery_node_demands[None, :], torch.ones_like(current_distance_matrix) * 1e6, torch.zeros_like(current_distance_matrix))
    pickup_penalty = torch.where(current_load_open[:, None] < pickup_node_demands[None, :], torch.ones_like(current_distance_matrix) * 1e6, torch.zeros_like(current_distance_matrix))
    
    # Combine heuristics
    combined_score = -torch.tanh(normalized_distance) - soft_time_penalty - hard_time_penalty - delivery_penalty - pickup_penalty
    
    # Introduce randomness to avoid local optima
    rand_weights = torch.rand_like(combined_score) * 0.1
    heuristic_scores = combined_score + rand_weights

    # Ensure scores are bounded and avoid extreme values
    heuristic_scores = torch.clamp(heuristic_scores, min=-1e3, max=1e3)
    
    return heuristic_scores