import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the computation related to 'current_distance_matrix', 'delivery_node_demands', and 'current_load'

    # New computation for distance-based heuristic scores
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # New computation for demand-based heuristic scores
    demand_scores_v2 = (current_load - delivery_node_demands.unsqueeze(0)) * 0.9 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.8

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    heuristic_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    return heuristic_scores_v2