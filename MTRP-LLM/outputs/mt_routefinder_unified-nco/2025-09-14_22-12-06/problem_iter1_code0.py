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
    
    # Initialize heuristic scores with negative infinity
    heuristic_scores = torch.full(current_distance_matrix.shape, float('-inf'))

    # Feasibility conditions
    can_deliver = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    can_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()
    
    # Time window feasibility
    earliest_arrival = arrival_times + current_distance_matrix
    within_time_window = ((earliest_arrival >= time_windows[:, 0].unsqueeze(0)) & 
                          (earliest_arrival <= time_windows[:, 1].unsqueeze(0))).float()
    
    # Route length condition
    within_length_limit = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine the three feasibility checks to find valid candidates
    valid_candidates = can_deliver * within_time_window * within_length_limit
    
    # Calculate heuristic scores based on distance and valid candidates
    distances = -current_distance_matrix * valid_candidates
    heuristic_scores = heuristic_scores + distances
    
    # Incorporate randomness to explore other options and avoid local optima
    random_noise = torch.randn_like(heuristic_scores) * 0.1
    heuristic_scores += random_noise

    # Normalize the heuristic scores to the range [0, 1]
    max_scores = heuristic_scores.max(dim=1, keepdim=True)[0]
    min_scores = heuristic_scores.min(dim=1, keepdim=True)[0]
    
    heuristic_scores = (heuristic_scores - min_scores) / (max_scores - min_scores + 1e-10)
    
    return heuristic_scores