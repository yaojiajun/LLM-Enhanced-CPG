import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic score matrix
    num_vehicles, num_nodes = current_distance_matrix.size()
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Compute distance-based heuristics (higher scores for shorter distances)
    inverse_distances = 1 / (current_distance_matrix + 1e-6)  # Prevent division by zero
    heuristic_scores += inverse_distances

    # Evaluate delivery constraints
    load_feasible = current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)
    heuristic_scores *= load_feasible.float()

    # Evaluate time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasible = (current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))
    heuristic_scores *= time_window_feasible.float()

    # Evaluate length constraints
    length_feasible = current_length.unsqueeze(1) >= current_distance_matrix
    heuristic_scores *= length_feasible.float()

    # Introduce randomness to avoid local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.05  # Adjust noise level as needed
    heuristic_scores += random_noise

    # Normalize scores to keep values bounded
    heuristic_scores = torch.clamp(heuristic_scores, min=-1, max=1)

    return heuristic_scores