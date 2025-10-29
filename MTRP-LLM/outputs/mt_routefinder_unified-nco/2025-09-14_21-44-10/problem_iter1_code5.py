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
    
    # Initialize the heuristic score matrix
    pomo_size, num_nodes = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)

    # Calculate feasible visits based on capacity, length, and time windows
    within_capacity = (
        (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) &
        (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    )
    
    current_time = arrival_times
    within_time_windows = (
        (current_time.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0)) &
        (current_time.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0))
    )
    
    within_length = (
        (current_length.unsqueeze(1) >= current_distance_matrix) 
    )

    # Combine feasibility conditions
    feasibility_mask = within_capacity & within_time_windows & within_length

    # Score adjustment based on distance, favoring shorter routes
    distance_scores = -current_distance_matrix.clone()  # Prefer lower distances

    # Randomness injection to avoid local optima
    randomness = torch.rand((pomo_size, num_nodes), device=current_distance_matrix.device) * 0.1

    # Apply feasibility mask, adding randomness to entries that are feasible
    heuristic_scores = torch.where(feasibility_mask, distance_scores + randomness, torch.tensor(float('-inf'), device=current_distance_matrix.device))

    return heuristic_scores