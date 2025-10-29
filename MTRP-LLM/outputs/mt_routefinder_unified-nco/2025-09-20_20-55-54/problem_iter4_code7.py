import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Introduce more diverse exploration using a combination of noise and distance-based scores
    noise = torch.randn_like(current_distance_matrix) * 1e-5
    perturbed_distances = current_distance_matrix + noise

    # Calculate heuristic score based on a combination of random noise and inverse distances with numerical stability
    heuristic_scores = (2 * torch.rand_like(current_distance_matrix) - 1) / (perturbed_distances + 1e-7)  # Introducing controlled randomness and numerical stability

    return heuristic_scores