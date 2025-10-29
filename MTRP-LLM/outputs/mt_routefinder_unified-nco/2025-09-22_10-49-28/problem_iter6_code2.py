import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix that increases penalty for longer distances with controlled randomness
    modified_distance_scores = -1.5 * (current_distance_matrix ** 2) + torch.randn_like(current_distance_matrix) * 0.3

    # Generate a modified demand-based heuristic score matrix by balancing current load and demand while introducing randomness
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.4 + (torch.min(current_load) / 5) + torch.randn_like(current_distance_matrix) * 0.35

    # Calculate the overall score matrix for edge selection by considering refined distance penalties and demand balancing
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores