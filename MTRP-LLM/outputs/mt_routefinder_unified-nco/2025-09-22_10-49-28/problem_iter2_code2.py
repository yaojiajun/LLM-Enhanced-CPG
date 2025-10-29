import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix that penalizes longer distances more severely
    modified_distance_scores = -1 * (current_distance_matrix ** 2) * 0.5 + torch.randn_like(current_distance_matrix) * 0.3

    # Generate a modified demand-based heuristic score matrix, prioritizing low-demand nodes with controlled randomness
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.4 + (torch.min(current_load) / 4) + torch.randn_like(current_distance_matrix) * 0.4

    # Calculate the overall score matrix for edge selection while retaining a proper scale
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores