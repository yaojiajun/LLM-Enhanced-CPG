import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Calculate potential scores based on distances and demand feasibility
    demand_scores = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * \
                    (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_scores = (arrival_times <= time_windows[:, 1].unsqueeze(0)).float() * \
                  (arrival_times >= time_windows[:, 0].unsqueeze(0)).float()

    # Length feasibility
    length_scores = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine scores with distance (minimizing distance)
    heuristic_scores = (1 - demand_scores) * (1 - time_scores) * (1 - length_scores) * current_distance_matrix
    
    # Introduce enhanced randomness to the scores
    random_factor = torch.rand_like(heuristic_scores) * 0.1  # Low random factor to ensure exploration
    heuristic_scores += random_factor
    
    # Normalize scores to keep them in a standard range
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min())
    
    return heuristic_scores