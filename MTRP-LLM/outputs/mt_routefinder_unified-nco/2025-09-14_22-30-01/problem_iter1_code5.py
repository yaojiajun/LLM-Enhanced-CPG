import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    pomo_size, N_plus_1 = current_distance_matrix.shape
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, N_plus_1), device=current_distance_matrix.device)

    # Calculate effective distances considering time windows, load, and length constraints
    time_penalty = (arrival_times - time_windows[:, 0].unsqueeze(0)).clamp(min=0)  # Penalty for arrival before time window
    time_score = -time_penalty  # Favorable to arrive as soon as possible

    # Check capacity constraints
    capacity_check = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()  # 1.0 if feasible, else 0.0
    capacity_check_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()  # For open route
    
    # Check current route duration limits
    duration_check = (current_length.unsqueeze(1) >= current_distance_matrix).float()  # 1.0 if feasible, else 0.0

    # Combine scores based on feasibility and distance
    feasible_mask = capacity_check * duration_check * (time_windows[:, 1].unsqueeze(0) >= arrival_times)  # Ensure all constraints are satisfied
    distance_scores = -current_distance_matrix  # Lower distance gives higher heuristic score
    
    # Calculate final heuristic scores
    heuristic_scores += feasible_mask * (distance_scores + time_score)  # Impact of distance and time on total score
    
    # Introduce randomness to avoid local optima
    random_scores = torch.rand((pomo_size, N_plus_1), device=current_distance_matrix.device) * 0.1  # Small random noise
    heuristic_scores += random_scores
    
    # Return the computed heuristic scores
    return heuristic_scores