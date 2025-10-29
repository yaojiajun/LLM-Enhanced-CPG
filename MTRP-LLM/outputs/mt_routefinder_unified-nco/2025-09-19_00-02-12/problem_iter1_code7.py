import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Randomly shuffle the distance matrix to introduce enhanced randomness
    shuffled_distance_matrix = current_distance_matrix.clone()
    for i in range(shuffled_distance_matrix.size(0)):
        shuffled_idx = torch.randperm(shuffled_distance_matrix.size(1))
        shuffled_distance_matrix[i] = shuffled_distance_matrix[i, shuffled_idx]

    # Introduce randomness by adding noise to the distance matrix
    noise = torch.rand_like(shuffled_distance_matrix) * 0.1  # Adjust noise level as needed
    noisy_distance_matrix = shuffled_distance_matrix + noise

    return noisy_distance_matrix