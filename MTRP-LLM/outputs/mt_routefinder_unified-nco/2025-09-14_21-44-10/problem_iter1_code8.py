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

    # Define weights for different criteria
    distance_weight = 0.5
    capacity_weight = 0.3
    time_window_weight = 0.2

    # Calculate distance scores (inverse of distances reducing cost for shorter distances)
    distance_scores = -current_distance_matrix * distance_weight

    # Capacity scores (positive for feasible deliveries, heavily penalizing over capacity)
    capacity_scores = torch.where((current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) &
                                   (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)),
                                   torch.zeros_like(current_distance_matrix, device=current_distance_matrix.device),
                                   -float('inf'))

    # Time window scores (penalizing delays outside time windows)
    current_time = arrival_times + current_distance_matrix
    time_window_scores = torch.where((current_time < time_windows[:, 0].unsqueeze(0)) | 
                                      (current_time > time_windows[:, 1].unsqueeze(0)),
                                      -float('inf'), 
                                      torch.zeros_like(current_distance_matrix, device=current_distance_matrix.device))

    # Length scores (penalizing routes exceeding duration limits)
    length_scores = torch.where(current_length.unsqueeze(1) >= current_distance_matrix.sum(dim=1, keepdim=True),
                                 torch.zeros_like(current_distance_matrix, device=current_distance_matrix.device),
                                 -float('inf'))

    # Calculate combined heuristic scores
    heuristic_scores += distance_scores + capacity_scores + time_window_scores + length_scores

    # Introduce randomness to avoid local optima
    random_adjustment = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += random_adjustment

    return heuristic_scores