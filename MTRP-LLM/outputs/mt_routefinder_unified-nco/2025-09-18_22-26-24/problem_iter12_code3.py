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
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate score based on distance, demands, load capacity, and time windows
    feasible_mask = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & \
                    (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)) & \
                    (current_length.unsqueeze(1) >= current_distance_matrix)
    
    # Favor shorter distances while penalizing based on demand and capacity issues
    distance_scores = -current_distance_matrix * feasible_mask.float()
    demand_penalty = (current_load.unsqueeze(1) < delivery_node_demands.unsqueeze(0)).float() * 1000
    demand_penalty_open = (current_load_open.unsqueeze(1) < delivery_node_demands_open.unsqueeze(0)).float() * 1000
    
    # Calculate scores based on time windows
    arrival_time_penalty = torch.clamp(arrival_times - time_windows[:, 0].unsqueeze(0), min=0)
    time_window_penalty = (arrival_time_penalty > 0).float() * 1000  # Penalize if outside time windows

    # Combine all scores
    heuristic_scores += distance_scores + demand_penalty + demand_penalty_open + time_window_penalty
    
    # Introduce randomness to help escape local optima
    random_noise = 0.05 * torch.randn_like(heuristic_scores)
    heuristic_scores += random_noise

    return heuristic_scores