import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the distance-based heuristic score matrix calculation
    distance_heuristic = torch.exp(-current_distance_matrix / torch.max(current_distance_matrix)) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Compute the demand-based heuristic score matrix with added noise for diversity
    demand_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce randomness with increased noise for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores for a diversified strategy
    cvrp_scores = distance_heuristic + demand_score + enhanced_noise

    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores remain the same as in the original function

    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores