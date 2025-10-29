import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the normalized distance-based heuristic score matrix with added diversity through randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Compute the demand-based heuristic score matrix with emphasis on high-demand nodes and enhanced randomness
    demand_scores = current_load * 0.8 + torch.max(delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores
    # Keep the existing calculations for these sections without modifications

    overall_scores=cvrp_scores+vrptw_scores+vrpb_scores+vrpl_scores+ovrp_scores

    return overall_scores