import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasibility masks for delivery and open routes
    delivery_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_length.unsqueeze(1) >= current_distance_matrix)
    open_route_feasible = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)) & (current_length.unsqueeze(1) >= current_distance_matrix)

    # Calculate time window feasibility
    time_window_feasible = (arrival_times.unsqueeze(1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)) & (arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0))

    # Scores based on feasibility
    heuristic_scores[delivery_feasible & time_window_feasible] += 1.0  # Positive score for feasible delivery routes
    heuristic_scores[open_route_feasible & time_window_feasible] += 1.0  # Positive score for feasible open routes
    heuristic_scores[~delivery_feasible & time_window_feasible] -= 0.5  # Slight penalty for infeasible delivery routes
    heuristic_scores[~open_route_feasible & time_window_feasible] -= 0.5  # Slight penalty for infeasible open routes

    # Incorporate distances into scores, prioritizing shorter routes
    heuristic_scores -= current_distance_matrix * 0.1  # Negative influence from distance

    # Add randomness to escape local optima
    random_noise = torch.randn_like(heuristic_scores) * 0.01
    heuristic_scores += random_noise

    return heuristic_scores