import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modify the calculation of heuristic scores for edge selection
    distance_heuristic = 1 / (current_distance_matrix + 1e-8)
    delivery_score = (1 / (delivery_node_demands + 1e-8)) * current_load
    pickup_score = (pickup_node_demands + 1e-8) / current_load

    # Introduce controlled randomness in scoring
    randomness = torch.rand_like(distance_heuristic)

    # Compute total score with a weighted sum of different heuristics
    total_score = 0.4 * distance_heuristic + 0.3 * delivery_score - 0.2 * pickup_score + 0.1 * randomness

    return total_score