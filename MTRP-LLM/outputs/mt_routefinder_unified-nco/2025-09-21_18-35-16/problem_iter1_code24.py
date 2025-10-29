import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Update the distance-based heuristic score matrix calculations with a different approach
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Update the demand-based heuristic score matrix calculations
    demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 1.2 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with a different noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores calculations remain unchanged from the original heuristics_v1 function
    
    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores