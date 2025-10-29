import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor,
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor,
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Constants
    epsilon = 1e-8

    # Distance Score: lower distance is better
    distance_scores = -current_distance_matrix

    # Capacity Score: penalize exceeding capacity
    capacity_mask = current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)
    capacity_scores = torch.where(capacity_mask, torch.zeros_like(current_distance_matrix), torch.full_like(current_distance_matrix, -1e6))

    # Time Window Score: penalize violations of time windows
    early_arrival = arrival_times < time_windows[:, 0].unsqueeze(0)
    late_arrival = arrival_times > time_windows[:, 1].unsqueeze(0)
    time_window_scores = torch.where(early_arrival | late_arrival, torch.full_like(current_distance_matrix, -1e6), torch.zeros_like(current_distance_matrix))

    # Length Budget Score: penalize routes exceeding length limits
    length_mask = current_length.unsqueeze(1) >= current_distance_matrix.sum(dim=1, keepdim=True)
    length_scores = torch.where(length_mask, torch.zeros_like(current_distance_matrix), torch.full_like(current_distance_matrix, -1e6))

    # Randomness for exploration
    random_scores = torch.rand_like(current_distance_matrix) * 0.1

    # Aggregate scores
    heuristic_scores = (distance_scores + capacity_scores + time_window_scores + length_scores + random_scores) / (4 + epsilon)

    # Mask invalid scores
    heuristic_scores[~torch.isfinite(heuristic_scores)] = -1e6  # Replace non-finite values with a large negative number

    return heuristic_scores