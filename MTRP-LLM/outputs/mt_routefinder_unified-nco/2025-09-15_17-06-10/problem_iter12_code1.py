import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Normalize distance matrix
    distance_max = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / distance_max
    
    # Calculate heuristic scores based on constraints
    rand_weights = torch.rand_like(current_distance_matrix)

    # Score components
    score_distance = torch.sigmoid(1 - normalized_distance) * rand_weights  # Prioritize closer nodes
    score_load = torch.sigmoid(current_load.unsqueeze(-1) - delivery_node_demands)  # Load feasibility
    score_time_window = torch.where((arrival_times < time_windows[:, 0].unsqueeze(0)), 
                                    (time_windows[:, 0].unsqueeze(0) - arrival_times) / 10, 
                                    torch.zeros_like(arrival_times))  # Penalize waiting time
    score_route_length = torch.sigmoid(current_length.unsqueeze(-1) - current_distance_matrix)  # Length feasibility

    # Combine scores for a final heuristic score
    heuristic_scores = score_distance + score_load + score_time_window + score_route_length

    # Incorporate randomness to avoid local optima
    enhanced_scores = heuristic_scores + 0.1 * (torch.rand_like(heuristic_scores) - 0.5)

    return enhanced_scores