import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)

    # Calculate delivery feasibility
    delivery_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))

    # Time window feasibility
    within_time_windows = (arrival_times.unsqueeze(1) <= time_windows[:, 1]) & (arrival_times.unsqueeze(1) >= time_windows[:, 0])

    # Current routing length feasibility
    route_length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)

    # Compute heuristic score based on the combination of the three feasibility constraints
    feasible_routes = delivery_feasible & within_time_windows & route_length_feasibility

    # Assign scores for feasible routes (lower distance, higher scores)
    score_matrix[feasible_routes] = 1 / (current_distance_matrix[feasible_routes] + 1e-6)  # Avoid division by zero

    # Add randomness to the scores for exploration
    randomness = torch.rand_like(score_matrix) * 0.1  # Small noise added
    score_matrix += randomness

    # Penalize undesirable edges (infeasible routes)
    score_matrix[~feasible_routes] = -torch.abs(current_distance_matrix[~feasible_routes])  # Assign negative scores

    return score_matrix