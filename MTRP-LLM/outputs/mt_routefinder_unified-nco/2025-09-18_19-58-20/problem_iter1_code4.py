import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Capacity feasibility
    delivery_feasibility = (delivery_node_demands <= current_load.unsqueeze(1)) & (delivery_node_demands_open <= current_load_open.unsqueeze(1))
    
    # Time window feasibility
    start_times = torch.maximum(arrival_times, time_windows[:, 0].unsqueeze(0))
    in_time_window = (start_times <= time_windows[:, 1].unsqueeze(0))

    # Route duration feasibility
    duration_feasibility = (current_distance_matrix < current_length.unsqueeze(1))

    # Combine feasibility checks
    feasible_edges = delivery_feasibility & in_time_window & duration_feasibility
    
    # Calculate heuristic scores for feasible edges
    heuristic_scores[feasible_edges] = -current_distance_matrix[feasible_edges]  # Favorable to select shorter paths

    # Penalize infeasible edges
    heuristic_scores[~feasible_edges] = current_distance_matrix[~feasible_edges] + 1000  # Adding a high cost for infeasibility

    # Introduce randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.5  # Random values in range [0, 0.5]
    heuristic_scores += randomness

    return heuristic_scores