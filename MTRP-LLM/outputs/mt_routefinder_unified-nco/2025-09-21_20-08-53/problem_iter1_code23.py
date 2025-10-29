import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Modify the normalized distance-based heuristic score matrix with added diversity through randomness
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score matrix with emphasis on low-demand nodes and enhanced randomness
    demand_scores_v2 = (torch.max(delivery_node_demands) - delivery_node_demands.unsqueeze(0)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce controlled randomness with noise level adjustment
    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores with diversified strategies for balanced exploration
    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    # Keep vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores calculation unchanged from the original function

    overall_scores_v2 = cvrp_scores_v2 + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores_v2