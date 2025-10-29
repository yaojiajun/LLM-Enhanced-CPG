import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Constraints evaluation
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    time_window_feasibility = (arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times > time_windows[:, 0].unsqueeze(0))
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)

    # Calculate scores based on feasible routes
    feasibility_mask = load_feasibility & time_window_feasibility & length_feasibility
    feasible_routes_distance = torch.where(feasibility_mask, current_distance_matrix, torch.tensor(float('inf')).to(current_distance_matrix.device))
    
    # Calculate inverse distance for scoring
    inverse_distance_scores = 1 / (feasible_routes_distance + 1e-6)  # avoid division by zero

    # Generate random noise to enhance exploration
    random_noise = torch.rand_like(inverse_distance_scores) * 0.1

    # Combine inverse distance scores with randomness
    heuristic_scores = inverse_distance_scores + random_noise

    # Normalize scores to avoid overflow and ensure they are in a useful range
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min() + 1e-6)

    # Assign negative scores for infeasible routes
    heuristic_scores[~feasibility_mask] = -1  # Assign negative scores to undesirable edges

    return heuristic_scores