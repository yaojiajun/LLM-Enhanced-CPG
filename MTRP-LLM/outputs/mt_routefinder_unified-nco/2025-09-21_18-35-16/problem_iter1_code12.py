import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Modified version of the normalized distance-based heuristic score matrix
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Compute the demand-based heuristic score matrix with adjusted weights and random noise
    demand_scores_v2 = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with moderate noise level
    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with modified strategies for balanced exploration
    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    # Include previous VRPTW, VRPB, VRPL, and OVRP calculations unaltered
    vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores = heuristics_v1(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length)

    overall_scores_v2 = cvrp_scores_v2 + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores_v2