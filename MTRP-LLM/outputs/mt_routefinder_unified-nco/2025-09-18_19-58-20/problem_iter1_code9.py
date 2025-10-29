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

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Check vehicle capacity constraints
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()

    # Check time window feasibility
    time_constraints = (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)).float() * \
                       (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float()

    # Check length constraints
    length_constraints = (current_length.unsqueeze(1) >= current_distance_matrix.sum(dim=1)).float()

    # Score assignments based on feasibility
    score_positive = feasible_delivery * feasible_pickup * time_constraints * length_constraints
    score_negative = (1 - feasible_delivery) + (1 - feasible_pickup) + (1 - time_constraints) + (1 - length_constraints)

    # Compute the heuristic scores as a blend of positive and negative scores
    heuristic_scores = score_positive - score_negative

    # Introduce randomness to enhance exploration
    randomness = torch.rand_like(heuristic_scores) * 0.1  # 10% random adjustment
    heuristic_scores += randomness

    return heuristic_scores