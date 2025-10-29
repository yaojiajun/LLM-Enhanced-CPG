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
    
    # Calculate feasibility based on capacity and current load
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_window_feasibility = ((arrival_times < time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times >= time_windows[:, 0].unsqueeze(0))).float()

    # Length constraint
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine feasibility for a score threshold
    feasibility_factor = delivery_feasibility * delivery_feasibility_open * time_window_feasibility * length_feasibility

    # Calculate heuristic scores based on distances and feasibilities
    # Higher distance impacts negatively, favoring shorter routes
    heuristic_scores += current_distance_matrix * (1 - feasibility_factor)

    # Introduce randomness to enhance exploration
    random_factors = torch.rand_like(heuristic_scores) * 0.1  # Small random perturbations
    heuristic_scores += random_factors
    
    # Assign scores based on exploration and prioritization
    # Favor routes that are feasible (lower distances rewarded with higher scores)
    heuristic_scores = -torch.where(feasibility_factor > 0, heuristic_scores, torch.tensor(float('inf')).to(heuristic_scores.device))

    return heuristic_scores