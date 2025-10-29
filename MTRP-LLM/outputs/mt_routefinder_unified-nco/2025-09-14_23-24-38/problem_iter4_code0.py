import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic scores with random values to enhance exploration
    heuristic_scores = torch.randn_like(current_distance_matrix) * 0.5
    
    # Calculate feasibility scores based on capacity constraints
    load_feasibility = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) >= 0
    load_feasibility_open = (current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)) >= 0
    
    # Calculate time window feasibility
    time_feasibility = (arrival_times.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0)) & (arrival_times.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0))
    
    # Consider remaining route length
    length_feasibility = (current_length.unsqueeze(1) - current_distance_matrix) >= 0
    
    # Compute combined feasibility scores
    feasibility_mask = load_feasibility & load_feasibility_open & time_feasibility & length_feasibility
    
    # Enhance scores in feasible directions and penalize infeasible options
    heuristic_scores += feasibility_mask.float() * 1.0  # Positive score for feasible routes
    heuristic_scores -= (~feasibility_mask).float() * 1.0  # Negative score for infeasible routes
    
    # Incorporate randomness for better exploration
    heuristic_scores += torch.rand_like(heuristic_scores) * 0.1 - 0.05
    
    return heuristic_scores