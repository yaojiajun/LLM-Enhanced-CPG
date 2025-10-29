import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8

    # Initialize heuristic score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)

    # Calculate load feasibility for capacity constraints
    load_remaining = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    feasible_load_mask = load_remaining >= 0

    # Calculate time feasibility for time window constraints
    time_service_windows = (arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times + current_distance_matrix < time_windows[:, 0].unsqueeze(0))
    feasible_time_mask = time_service_windows

    # Calculate route length feasibility for duration limits
    route_length_remaining = current_length.unsqueeze(1) - current_distance_matrix
    feasible_length_mask = route_length_remaining >= 0

    # Calculate overall feasibility
    feasibility_mask = feasible_load_mask & feasible_time_mask & feasible_length_mask

    # Score calculation
    scoring_factor = 1 / (current_distance_matrix + epsilon)
    score_matrix[feasibility_mask] = scoring_factor[feasibility_mask]  # Positive scores for feasible edges
    
    # Penalize infeasible edges with negative scores
    score_matrix[~feasibility_mask] = -1 / (current_distance_matrix[~feasibility_mask] + epsilon)

    # Controlled randomness for exploration
    randomness = torch.rand_like(score_matrix) * 0.1  # Small noise
    score_matrix += randomness

    # Clamp scores to ensure they are finite
    score_matrix = torch.clamp(score_matrix, min=-1e10, max=1e10)
    
    return score_matrix