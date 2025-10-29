import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Generate a new modified distance-based heuristic score matrix with customized exponent and weight
    distance_heuristic = -1 * (current_distance_matrix ** 1.7) * 0.08 + torch.randn_like(current_distance_matrix) * 0.1

    # Generate the modified demand-based heuristic score matrix with refined balancing and weight
    delivery_score = (current_load.unsqueeze(1) - 1.5*delivery_node_demands.unsqueeze(0)).float() * 0.6 + torch.randn_like(current_distance_matrix) * 0.1

    # Calculate the overall score matrix for edge selection with updated calculations
    overall_scores = distance_heuristic + delivery_score

    return overall_scores