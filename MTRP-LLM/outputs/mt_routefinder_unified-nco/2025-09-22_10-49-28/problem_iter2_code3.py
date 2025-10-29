import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a new heuristic score matrix with modified calculations for distance, delivery demand, and current load
    distance_heuristic = current_distance_matrix * 0.6 + torch.randn_like(current_distance_matrix) * 0.2

    delivery_score = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.7 + torch.min(current_load) / 3 + torch.randn_like(current_distance_matrix) * 0.6

    pickup_score = (current_load.unsqueeze(1) - pickup_node_demands.unsqueeze(0)) * 0.5 + torch.min(current_load) / 5 + torch.randn_like(current_distance_matrix) * 0.3

    # Calculate the overall score matrix for edge selection
    overall_scores = distance_heuristic + delivery_score + pickup_score

    return overall_scores