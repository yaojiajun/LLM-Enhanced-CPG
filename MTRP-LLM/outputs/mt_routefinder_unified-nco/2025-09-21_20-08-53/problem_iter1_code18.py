import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Adjusted heuristic computations for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    # Compute the normalized distance-based heuristic score matrix with different randomness factor
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Compute the demand-based heuristic score matrix with varied emphasis on high-demand nodes and randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased noise for exploration with a different noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the adjusted heuristic scores with varied strategies
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Retain the implementation for other computations as in heuristics_v1

    return cvrp_scores