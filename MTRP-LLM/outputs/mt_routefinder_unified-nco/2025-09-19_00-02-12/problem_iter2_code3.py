import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Compute time window satisfaction
    earliest_arrival = arrival_times + current_distance_matrix
    time_window_satisfaction = (earliest_arrival >= time_windows[:, 0]) & (earliest_arrival <= time_windows[:, 1])
    
    # Incorporate delivery demands and capacity constraints
    capacity_ok = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0))
    capacity_ok_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Incorporate load and time window feasibility into heuristic scores
    heuristic_scores[capacity_ok & time_window_satisfaction] += 1.0  # Positive score for feasible nodes
    
    # Penalize nodes that exceed capacity or are outside of time windows
    heuristic_scores[~capacity_ok & time_window_satisfaction] -= 1.0
    heuristic_scores[capacity_ok & ~time_window_satisfaction] -= 1.0
    
    # Random weight to enhance exploration
    randomness = (torch.rand_like(current_distance_matrix) * 0.5) - 0.25
    heuristic_scores += randomness
    
    # Normalize heuristic scores to ensure they stay within a range
    heuristic_scores = torch.clamp(heuristic_scores, min=-2.0, max=2.0)
    
    return heuristic_scores