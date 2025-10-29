import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Constants
    epsilon = 1e-8
    
    # Calculate feasibility masks
    delivery_mask = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    open_delivery_mask = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    time_windows_mask = (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float() * \
                        (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)).float()
    
    length_mask = (current_length.unsqueeze(-1) >= current_distance_matrix).float()
    
    # Heuristic scores based on distance and feasibility
    feasibility_score = delivery_mask * open_delivery_mask * time_windows_mask * length_mask
    
    # Calculate normalized distances
    normalized_distances = 1.0 / (current_distance_matrix + epsilon)
    clamped_distances = torch.clamp(normalized_distances, min=0, max=10)  # clamp to avoid extremes
    
    # Heuristic score matrix
    heuristic_scores = feasibility_score * clamped_distances
    
    # Introducing randomness
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Controlled randomness
    heuristic_scores += randomness
    
    # Final output ensuring all values are finite
    heuristic_scores = torch.clamp(heuristic_scores, min=float('-inf'), max=float('inf'))
    
    return heuristic_scores