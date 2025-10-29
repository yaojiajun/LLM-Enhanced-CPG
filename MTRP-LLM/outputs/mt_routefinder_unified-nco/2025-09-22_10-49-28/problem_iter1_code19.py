import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the heuristics calculation for distance matrix, delivery demands, and current load
    # Compute the normalized distance-based heuristic score matrix with added diversity through randomness
    distance_scores = current_distance_matrix / torch.max(current_distance_matrix) - torch.randn_like(
        current_distance_matrix) * 0.7

    # Compute the modified demand-based heuristic score matrix with emphasis on low-demand nodes and added randomness
    demand_scores = (torch.max(delivery_node_demands) - delivery_node_demands.unsqueeze(0)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    total_scores = distance_scores + demand_scores

    # Include additional randomness for exploration
    total_scores += torch.randn_like(current_distance_matrix) * 0.3

    return total_scores