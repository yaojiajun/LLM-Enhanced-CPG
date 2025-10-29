import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modify the normalized distance-based heuristic score computation for 'current_distance_matrix'
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 1.5

    # Modify the demand-based heuristic score calculation for 'delivery_node_demands' and 'current_load'
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.7 + torch.max(
        delivery_node_demands) / 4 + torch.randn_like(current_distance_matrix) * 0.2

    # Retain the same noise level for diversity and exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the updated heuristic scores for improved edge selection
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    return cvrp_scores