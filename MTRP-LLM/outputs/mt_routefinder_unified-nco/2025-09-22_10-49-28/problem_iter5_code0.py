import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a mutated distance and demand-based heuristic score matrix with refined penalty weights and controlled randomness
    distance_heuristic = -1 * (current_distance_matrix ** 1.3) * 0.18 + torch.randn_like(current_distance_matrix) * 0.3
    delivery_score = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) ** 2 * 0.55 + torch.randn_like(current_distance_matrix) * 0.4
    pickup_score = (current_load.unsqueeze(1) - pickup_node_demands.unsqueeze(0).float()) * 0.3 + torch.randn_like(current_distance_matrix) * 0.2

    # Calculate the overall score matrix for edge selection with adjusted weights
    overall_scores = distance_heuristic + delivery_score + pickup_score

    return overall_scores