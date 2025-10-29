import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Change the way distance-based heuristic score is computed
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7 + torch.exp(-current_distance_matrix) * 0.5

    # Retain the original demand-based heuristic score calculation
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores + enhanced_noise

    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores - Same calculations as in heuristics_v1
    # ...

    overall_scores_v2 = cvrp_scores_v2 + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores_v2