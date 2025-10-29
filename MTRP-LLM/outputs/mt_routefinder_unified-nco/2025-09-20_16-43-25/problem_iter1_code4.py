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

    # Shape constants
    pomo_size, N_plus_one = current_distance_matrix.shape
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, N_plus_one), device=current_distance_matrix.device)

    # Capacity feasibility checks
    delivery_capacity = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    open_capacity = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Time window feasibility checks
    arrival_within_windows = ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)) & \
                             ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0))
    
    # Duration feasibility checks
    within_duration = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine feasibility checks
    feasibility_mask = (delivery_capacity * open_capacity * arrival_within_windows.float() * within_duration)

    # Calculate heuristic indicators
    # Higher penalty for distance to promote shorter paths, adjusted by feasibility
    distance_penalty = (1 / (current_distance_matrix + 1e-6)) * feasibility_mask  # Avoid division by zero
    heuristic_scores += distance_penalty

    # Randomness to encourage exploration of solution space
    random_exploration = torch.rand_like(heuristic_scores) * 0.1  # Small random component
    heuristic_scores += random_exploration

    # Penalize undesirable edges (non-feasible) by setting their score to a large negative value
    heuristic_scores[feasibility_mask == 0] -= 1e6  # Large negative score for infeasible edges

    return heuristic_scores