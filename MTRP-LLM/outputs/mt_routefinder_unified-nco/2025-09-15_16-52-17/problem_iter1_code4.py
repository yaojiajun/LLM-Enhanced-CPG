import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate the feasibility masks for delivery and pickup demands
    delivery_feasibility = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Check time window feasibility
    time_feasibility = ((arrival_times.unsqueeze(-1) + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float()
    time_feasibility *= (arrival_times.unsqueeze(-1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float()
   
    # Evaluate remaining length feasibility
    length_feasibility = (current_length.unsqueeze(-1) >= current_distance_matrix).float()
    
    # Combined feasibility score
    feasibility_score = delivery_feasibility * delivery_feasibility_open * time_feasibility * length_feasibility
    
    # Calculate heuristic cost with penalties for infeasibility
    penalty_factor = 1.5  # Adjustable parameter for penalizing infeasible edges
    cost_scores = current_distance_matrix * (1.0 - feasibility_score * penalty_factor)
    
    # Incorporate randomness to enhance exploration, avoiding local optima
    random_factors = torch.rand_like(cost_scores) * 0.1  # Slight random perturbation
    heuristic_scores = cost_scores + random_factors
    
    # Assign negative scores to infeasible edges to discourage selection
    heuristic_scores[feasibility_score == 0] -= 1000  # Large penalty for infeasibility

    return heuristic_scores