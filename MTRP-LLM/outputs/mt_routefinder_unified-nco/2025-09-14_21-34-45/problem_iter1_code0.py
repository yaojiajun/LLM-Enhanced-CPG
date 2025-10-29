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

    # Initialize the heuristic score matrix
    pomo_size, n_nodes = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, n_nodes), device=current_distance_matrix.device)

    # Calculate remaining capacity and time window feasibility
    capacity_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * \
                        (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    time_window_feasible = ((arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)).float() *
                             (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float())
    
    length_feasible = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine feasibility criteria
    feasibility_mask = capacity_feasible * time_window_feasible * length_feasible

    # Compute heuristic based on distance and feasibility
    distance_scores = -current_distance_matrix * feasibility_mask  # Penalty for distance if feasible

    # Randomness addition to avoid local optima
    randomness = torch.rand_like(distance_scores) * (1 - feasibility_mask)  # only add when not feasible
    heuristic_scores = distance_scores + randomness

    # Apply additional scoring for pickups and mixed demands
    pickup_bonus = (pickup_node_demands.unsqueeze(0) > 0).float() * 10  # Encourage pickups
    heuristic_scores += pickup_bonus * feasibility_mask  # Only add bonus where feasible

    return heuristic_scores