import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix emphasizing more on shorter distances and controlled randomness
    modified_distance_scores = -(current_distance_matrix ** 0.8) * 0.4 + torch.randn_like(current_distance_matrix) * 0.3

    # Retain the original demand-based heuristic score matrix calculation
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + (torch.max(delivery_node_demands) / 3) + torch.randn_like(current_distance_matrix) * 0.3

    # Combine the modified distance and demand scores for final heuristic scoring
    overall_scores = modified_distance_scores + demand_scores

    return overall_scores