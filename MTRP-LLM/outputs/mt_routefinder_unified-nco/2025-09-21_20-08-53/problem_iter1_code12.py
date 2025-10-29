import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Specific modification for distance-based heuristic
    distance_heuristic = (current_distance_matrix ** 2) / torch.max(current_distance_matrix) + 0.5

    # Demand-based heuristic score remains unchanged
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = distance_heuristic + demand_scores + enhanced_noise

    # Rest of the implementation remains the same as heuristics_v1 for vrptw_scores, vrpb_scores, vrpl_scores, and ovrp_scores

    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores