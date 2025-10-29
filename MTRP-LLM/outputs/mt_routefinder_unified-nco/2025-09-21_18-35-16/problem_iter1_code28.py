import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the heuristic score matrix for distance with modified randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Compute the demand-based heuristic score matrix with adjusted weights and noise levels
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.7 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce increased randomness for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores calculations remain the same as in heuristics_v1

    overall_scores=cvrp_scores+vrptw_scores+vrpb_scores+vrpl_scores+ovrp_scores

    return overall_scores