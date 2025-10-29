import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasible routes based on capacity and length constraints
    capacity_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    capacity_feasible_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    length_feasible = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Time window constraints: Calculate waiting times and feasibility
    earliest_arrival = arrival_times + current_distance_matrix
    within_time_windows = ((earliest_arrival >= time_windows[:, 0].unsqueeze(0)) & (earliest_arrival <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Compute additional scores based on all constraints
    feasibility = capacity_feasible * capacity_feasible_open * length_feasible * within_time_windows
    
    # Calculate costs: An inverse transformation to prioritize shorter distances and penalties for longer routes
    cost_scores = 1 / (1 + current_distance_matrix)
    
    # Combine the feasibility with negative costs to derive heuristic scores
    heuristic_scores = (feasibility * cost_scores) - (1 - feasibility) * 10  # Heavily penalize infeasible routes

    # Introduce enhanced randomness to encourage exploration and avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Adjust the scale of randomness as needed
    heuristic_scores += randomness
    
    return heuristic_scores