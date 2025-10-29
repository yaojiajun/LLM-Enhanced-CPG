import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Reversed normalized distance-based heuristic score matrix with controlled randomness
    reversed_distance_scores = current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Generate a new delivery-based heuristic score matrix with different emphasis
    delivery_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce distinct randomness for exploration with varied noise levels
    diverse_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores with unique strategies for improved exploration
    modified_scores = reversed_distance_scores + delivery_scores + diverse_noise

    # Keep the rest of the original calculations unchanged

    return modified_scores