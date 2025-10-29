import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix where longer distances are penalized slightly less severely
    modified_distance_scores = -1 * ((current_distance_matrix ** 2) * 0.4) + torch.randn_like(current_distance_matrix) * 0.3

    # Generate a modified demand-based heuristic score matrix by adding a combination of delivery and pickup demands with tailored randomness
    demand_scores = (current_load.repeat(current_distance_matrix.shape[1], 1).t() + delivery_node_demands.unsqueeze(0) + (pickup_node_demands * 0.8).unsqueeze(0)) * 0.5
    modified_demand_scores = demand_scores - torch.min(demand_scores) + torch.randn_like(current_distance_matrix) * 0.2

    # Calculate the overall score matrix for edge selection while maintaining a proper scale
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores