import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix with a softened penalization curve
    modified_distance_scores = -0.3 * (current_distance_matrix ** 1.8) + torch.randn_like(current_distance_matrix) * 0.25

    # Generate a modified demand-based heuristic score matrix emphasizing high-demand nodes with adjusted randomness
    modified_demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + (torch.max(delivery_node_demands) / 4) + torch.randn_like(current_distance_matrix) * 0.3

    # Calculate the overall score matrix for edge selection while retaining a proper scale
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores