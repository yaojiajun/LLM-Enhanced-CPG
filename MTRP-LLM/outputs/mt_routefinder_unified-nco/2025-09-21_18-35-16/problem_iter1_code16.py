import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Updated heuristics based on the modified calculation for current_distance_matrix, delivery_node_demands, and current_load
    # Distance-based heuristic score matrix with modified normalization and randomness
    normalized_distance_scores = -current_distance_matrix / (torch.max(current_distance_matrix) + 1e-8) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Demand-based heuristic score matrix with adjusted demand influence and randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 1.0 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with moderate noise levels
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with modified strategies for balanced exploration
    overall_scores = normalized_distance_scores + demand_scores + enhanced_noise

    return overall_scores