import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the distance-based heuristic score matrix calculation with squared distances and adjusted normalization
    normalized_distance_scores = -(current_distance_matrix ** 2) / torch.max(current_distance_matrix ** 2) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score matrix calculation with a different formula
    demand_scores = (delivery_node_demands.unsqueeze(0) + current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) * 0.3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce increased randomness for exploration with a different noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep the rest of the function implementation unchanged
    return cvrp_scores