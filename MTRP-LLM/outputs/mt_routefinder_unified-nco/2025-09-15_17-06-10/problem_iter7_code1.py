import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Generate random weights with a more dynamic range
    rand_weights = torch.rand_like(current_distance_matrix) * 10 - 5  # weights from -5 to 5

    # Normalize the distance matrix to prevent large values dominating the score
    max_distance = torch.max(current_distance_matrix, dim=1, keepdim=True).values
    normalized_distance = current_distance_matrix / (max_distance + 1e-6)  # Avoiding division by zero

    # Compute scores based on different heuristics
    score_distance = torch.sigmoid(1 - normalized_distance) * rand_weights
    score_capacity = torch.tanh(current_load.unsqueeze(1) / (delivery_node_demands + 1e-6))  # Avoiding division by zero
    score_time_window = torch.where((arrival_times < time_windows[:, 0].unsqueeze(0)), 0, 
                                    torch.where((arrival_times > time_windows[:, 1].unsqueeze(0)), -1, 1))  # -1 if late, 0 if early, 1 if on time
    
    # Penalty for not having enough capacity for delivery or pickup
    capacity_penalty = -(delivery_node_demands > current_load.unsqueeze(1)).float() * 1000  # Heavy penalty

    # Combine scores with nonlinear transformations for more exploration
    heuristic_scores = (score_distance + score_capacity + score_time_window + capacity_penalty) / 4.0  # Normalize the score
    
    # Add a mix of randomness to avoid convergence to local optima
    heuristic_scores += (torch.rand_like(heuristic_scores) - 0.5) * 0.1  # Random noise

    return heuristic_scores