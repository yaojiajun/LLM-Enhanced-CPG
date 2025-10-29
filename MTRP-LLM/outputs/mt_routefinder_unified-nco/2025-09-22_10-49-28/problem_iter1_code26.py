import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Modified computation of normalized distance-based heuristic score matrix with diverse randomness
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Modified demand-based heuristic score matrix with emphasis on demand ratio and randomness
    demand_scores_v2 = (delivery_node_demands / (current_load + 1e-6)) * 1.1 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce increased randomness for exploration with controlled noise level
    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with modified strategies for balanced exploration
    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    # Keep calculations for other constraints unchanged
    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores

    return cvrp_scores_v2 + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores