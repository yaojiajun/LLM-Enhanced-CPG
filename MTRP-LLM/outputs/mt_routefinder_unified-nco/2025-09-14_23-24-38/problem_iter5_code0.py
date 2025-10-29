import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic scores matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Constraints evaluation
    capacity_constraints = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    capacity_constraints_open = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window constraints
    now = arrival_times + current_distance_matrix
    time_window_feasibility = ((now >= time_windows[:, 0].unsqueeze(0)) & (now <= time_windows[:, 1].unsqueeze(0))).float()

    # Length constraints
    length_constraints = (current_length.unsqueeze(-1) >= current_distance_matrix).float()

    # Combined scores based on constraints application
    feasibility = capacity_constraints * capacity_constraints_open * time_window_feasibility * length_constraints

    # Compute heuristic score based on distance, prioritized by feasibility
    heuristic_scores = feasibility * (1 / (current_distance_matrix + 1e-6))  # Adding small epsilon to avoid division by zero

    # Introduce randomness based on constraint severity
    random_factor = torch.rand_like(heuristic_scores) * (1 - feasibility)  # Introduce exploration where not feasible
    heuristic_scores += random_factor * 0.2  # Scaled random term to allow exploration

    # Normalize scores to prevent overwhelming influence of one node
    heuristic_scores = heuristic_scores / (torch.sum(heuristic_scores, dim=1, keepdim=True) + 1e-6)

    return heuristic_scores