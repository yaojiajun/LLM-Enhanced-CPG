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
    
    # Define constants for weighting heuristic components
    weight_distance = 0.5
    weight_time_window = 0.3
    weight_capacity = 0.2
    
    # Compute distances and initialize heuristic scores
    heuristic_scores = -current_distance_matrix.clone()
    
    # Time window feasibility scores (add large penalty if out of window)
    earliest_arrival = arrival_times + current_distance_matrix
    latest_arrival = time_windows[:, 1].unsqueeze(0)  # Broadcasting

    time_window_feasibility = (earliest_arrival <= latest_arrival).float()
    time_window_penalty = (earliest_arrival > time_windows[:, 1].unsqueeze(0)).sum(dim=1, keepdim=True).float() * -1000.0
    heuristic_scores += weight_time_window * time_window_feasibility + time_window_penalty

    # Capacity feasibility scores
    capacity_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    capacity_penalty = (current_load_open.unsqueeze(1) < delivery_node_demands_open.unsqueeze(0)).float() * -100.0
    heuristic_scores += weight_capacity * capacity_feasibility + capacity_penalty
    
    # Adjust scores based on remaining route length
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    heuristic_scores += length_feasibility
  
    # Introduce randomness to avoid convergence to local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.01
    heuristic_scores += random_noise

    # Normalize heuristic scores to [0, 1] for better interpretability
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min() + 1e-6)
    
    return heuristic_scores