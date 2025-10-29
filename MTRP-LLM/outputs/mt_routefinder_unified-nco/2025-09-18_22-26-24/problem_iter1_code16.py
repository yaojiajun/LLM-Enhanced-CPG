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
    
    num_nodes = delivery_node_demands.size(0)
    num_trajectories = current_load.size(0)
    
    # Initialize the score matrix
    heuristic_scores = torch.zeros((num_trajectories, num_nodes), device=current_distance_matrix.device)
    
    # Calculate distance factors
    distance_scores = -current_distance_matrix  # Negative because shorter distances should get higher scores
    
    # Capacity constraints for delivery demands
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    time_window_scores = torch.zeros((num_trajectories, num_nodes), device=current_distance_matrix.device)
    for i in range(num_nodes):
        earliest, latest = time_windows[i]
        arrival_time = arrival_times[:, i]
        is_within_time_window = (arrival_time >= earliest) & (arrival_time <= latest)
        time_window_scores[:, i] = is_within_time_window.float()
    
    # Route length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Pickup demand feasibility
    pickup_feasibility = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= current_load.max()).float()
    
    # Combine all factors into heuristic scores
    heuristic_scores += distance_scores + 10 * delivery_feasibility + 5 * time_window_scores + 8 * length_feasibility + 7 * pickup_feasibility
    
    # Introduce randomness to avoid local optima
    random_noise = torch.normal(0, 0.1, size=heuristic_scores.shape, device=current_distance_matrix.device)
    heuristic_scores += random_noise
    
    return heuristic_scores