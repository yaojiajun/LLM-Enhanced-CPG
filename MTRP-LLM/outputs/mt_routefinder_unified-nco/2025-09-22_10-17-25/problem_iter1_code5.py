import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristic calculation for current_distance_matrix
    normalized_distance_scores = -torch.sqrt(current_distance_matrix) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Modified demand-based heuristic calculation
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    overall_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remaining parts of the original function unchanged
    earliest_times = time_windows[:, 0].unsqueeze(0)
    latest_times = time_windows[:, 1].unsqueeze(0)
    arrival_times = arrival_times
    late_arrivals = torch.clamp(arrival_times - latest_times, min=0)

    # Final output
    return overall_scores