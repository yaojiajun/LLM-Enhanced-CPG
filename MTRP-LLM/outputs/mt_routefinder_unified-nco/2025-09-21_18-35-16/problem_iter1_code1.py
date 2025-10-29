import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score matrix with increased diversity through randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Modify the demand-based heuristic score matrix with different emphasis and enhanced randomness
    demand_scores = (2 * delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.5 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce even higher randomness and variability for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 3.0

    # Combine the modified heuristic scores with diverse strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep the rest of the code unchanged
    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores computations...

    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores