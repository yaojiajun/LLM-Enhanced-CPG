import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix that incorporates additional penalty for longer distances
    modified_distance_scores = -1 * (current_distance_matrix ** 2) * 0.6 + torch.randn_like(current_distance_matrix) * 0.3

    # Generate a modified demand-based heuristic score matrix by emphasizing current load balance and random fluctuations
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.5 + (torch.min(current_load) / 4) + torch.randn_like(current_distance_matrix) * 0.4

    # Calculate the overall score matrix for edge selection while considering distance penalties and demand priorities
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores