import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a mutated distance-based heuristic score matrix with adjusted penalty weights and controlled randomness
    mutated_distance_scores = -1 * (current_distance_matrix ** 1.5) * 0.1 + torch.randn_like(current_distance_matrix) * 0.2

    # Generate the modified demand-based heuristic score matrix with altered demand balance calculation and randomness
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.2 + (1 / (1 + torch.exp(torch.mean(current_load) - torch.min(current_load)))) + torch.randn_like(current_distance_matrix) * 0.3

    # Calculate the overall score matrix for edge selection with adjusted weights and scaling
    overall_scores = mutated_distance_scores + modified_demand_scores

    return overall_scores