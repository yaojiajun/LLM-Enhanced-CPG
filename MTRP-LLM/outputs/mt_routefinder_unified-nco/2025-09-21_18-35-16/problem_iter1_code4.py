import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristic computation for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    # Compute the normalized distance-based heuristic score matrix with added diversity through randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Compute the demand-based heuristic score matrix with adjustments and randomness
    demand_scores = (3 * delivery_node_demands.unsqueeze(0) - 2 * current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    modified_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Now keep the rest of the code as is
    # vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores, overall_scores calculation
    
    return overall_scores