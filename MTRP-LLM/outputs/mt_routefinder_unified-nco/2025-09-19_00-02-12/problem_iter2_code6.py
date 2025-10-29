import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasible deliveries based on load constraints
    load_feasibility = current_load.unsqueeze(1) >= delivery_node_demands
    load_feasibility_open = current_load_open.unsqueeze(1) >= delivery_node_demands_open
    effective_load_feasibility = load_feasibility & load_feasibility_open

    # Adjust scores based on distance: closer nodes should have higher scores
    distance_score = 1 / (current_distance_matrix + 1e-6)  # Avoid division by zero
    heuristic_scores += effective_load_feasibility.float() * distance_score

    # Evaluate time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasibility = (current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))
    heuristic_scores += time_window_feasibility.float() * 0.5  # Weight time window feasibility

    # Penalize routes that exceed remaining length budget
    length_penalty = current_length.unsqueeze(1) - current_distance_matrix
    length_penalty[length_penalty < 0] = -torch.abs(length_penalty[length_penalty < 0])  # Negative for infeasible
    heuristic_scores += length_penalty

    # Introduce some randomness to avoid local optima
    randomness_factor = torch.rand_like(heuristic_scores) * 0.1  # Small random perturbation
    heuristic_scores += randomness_factor

    return heuristic_scores