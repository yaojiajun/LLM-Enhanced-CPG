import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    num_vehicles, num_nodes = current_distance_matrix.shape
    
    # Calculate the effective capacity remaining for deliveries
    effective_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    effective_capacity_open = current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)

    # Calculate the time window penalties
    earliest_service = time_windows[:, 0].unsqueeze(0)
    latest_service = time_windows[:, 1].unsqueeze(0)
    
    # Estimating time windows violations
    time_window_penalty = (arrival_times.unsqueeze(1) < earliest_service).float() * float('inf') \
                          + (arrival_times.unsqueeze(1) > latest_service).float() * float('inf')

    # Calculate route duration limits penalty
    duration_penalty = (current_length.unsqueeze(1) + current_distance_matrix) > 1e-8
    duration_limit_mask = duration_penalty.float() * float('inf')
    
    # Combine penalties for infeasible solutions
    penalties = time_window_penalty + duration_limit_mask

    # Calculate delivery score based on distance and capacity
    delivery_scores = current_distance_matrix / (effective_capacity + epsilon)

    # Calculate delivery scores for open routes
    open_route_scores = current_distance_matrix / (effective_capacity_open + epsilon)
    
    # Combine scores and apply penalties
    heuristic_scores = torch.where(penalties < float('inf'), delivery_scores - penalties, -torch.inf)
    open_route_scores_masked = torch.where(penalties < float('inf'), open_route_scores - penalties, -torch.inf)

    # Introduce randomness to avoid premature convergence
    randomness = (torch.rand_like(heuristic_scores) * 0.1) - 0.05  # Random values in [-0.05, 0.05]
    heuristic_scores += randomness

    # Ensure that scores are finite
    heuristic_scores = torch.clamp(heuristic_scores, min=-1e10, max=1e10)

    return heuristic_scores