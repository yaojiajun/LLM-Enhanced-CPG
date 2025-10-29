import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Compute heuristic scores based on various conditions and constraints
    heuristic_scores = (current_distance_matrix / (delivery_node_demands.unsqueeze(0) + 1e-8))  # Example heuristic score calculation
    
    # Introduce randomness to avoid local optima
    noise = torch.rand(heuristic_scores.size()) * 1e-6
    heuristic_scores += noise
    
    return heuristic_scores