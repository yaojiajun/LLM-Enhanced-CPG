import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.full(current_distance_matrix.shape, -float('inf'), device=current_distance_matrix.device)
    
    # Calculate feasibilities
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    time_feasibility = (arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times + current_distance_matrix < time_windows[:, 1].unsqueeze(0))
    time_feasibility = time_feasibility.float() * (arrival_times >= time_windows[:, 0].unsqueeze(0)).float()
    
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Aggregate feasibility checks
    feasibility = delivery_feasibility * delivery_feasibility_open * time_feasibility * length_feasibility
    
    # Calculate heuristic scores based on distances, weighted by feasibility
    distance_scores = -current_distance_matrix * feasibility  # Negative distances to promote shorter routes
    
    # Introduce randomness to avoid local optima
    random_noise = torch.rand_like(distance_scores) * 0.1  # Slight random noise
    heuristic_scores = distance_scores + random_noise
    
    # Applying a cap on heuristic scores based on certain criteria (e.g., max distance)
    cap_threshold = -current_distance_matrix.mean()  # Mean of distances to cap the scores
    heuristic_scores = torch.where(heuristic_scores < cap_threshold, heuristic_scores, torch.tensor(float('inf'), device=current_distance_matrix.device))
    
    # Final scores with positive promoting feasible nodes and negative for otherwise
    return heuristic_scores