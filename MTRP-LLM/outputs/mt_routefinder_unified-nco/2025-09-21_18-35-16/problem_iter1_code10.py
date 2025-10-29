import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Compute a modified version of normalized distance-based heuristic score matrix with added randomness
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Compute a modified version of demand-based heuristic score matrix with increased randomness
    delivery_scores_v2 = 2.0 * (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Combine the modified heuristic scores with diversified strategies
    cvrp_scores_v2 = normalized_distance_scores_v2 + delivery_scores_v2

    # Keep the original computation for other heuristic components

    # Remaining parts of the function unchanged from v1