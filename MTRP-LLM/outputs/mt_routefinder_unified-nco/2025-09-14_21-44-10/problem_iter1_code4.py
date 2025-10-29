import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Heuristic score matrix initialization
    score_matrix = torch.zeros_like(current_distance_matrix)
    
    # Capacity constraints
    capacity_mask = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    open_capacity_mask = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window constraints
    time_window_mask = ((arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) &
                        (arrival_times.unsqueeze(1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0))).float()
    
    # Length constraints
    length_mask = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine all constraints
    feasibility_mask = capacity_mask * open_capacity_mask * time_window_mask * length_mask
    
    # Calculate heuristic scores based on distance and feasibility
    heuristic_scores = torch.where(feasibility_mask > 0, 
                                    1 / (current_distance_matrix + 1e-6),  # Prioritize shorter distances
                                    torch.tensor(float('-inf')).to(current_distance_matrix.device))  # Assign negative score to infeasible edges

    # Enhance randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Small random values to add variability
    heuristic_scores += randomness

    # Final score matrix calculation
    score_matrix = heuristic_scores * feasibility_mask
    
    return score_matrix