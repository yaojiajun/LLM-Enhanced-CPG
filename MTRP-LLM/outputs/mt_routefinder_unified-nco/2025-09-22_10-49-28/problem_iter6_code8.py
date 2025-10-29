import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a modified distance-based heuristic score matrix with a different penalty curve and randomness
    modified_distance_scores = -1 * (current_distance_matrix ** 1.5) * 0.2 + (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.8 + torch.randn_like(current_distance_matrix) * 0.15

    # Use the original demand-based heuristic score matrix calculation
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.35 + (1 / (1 + torch.exp(torch.mean(current_load) - torch.min(current_load))) + torch.randn_like(current_distance_matrix) * 0.3)
    
    # Calculate the overall score matrix for edge selection with appropriate scaling
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores