import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize distance matrix
    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / max_distance if max_distance > 0 else current_distance_matrix

    # Dynamic random weights to promote exploration
    rand_weights = torch.rand_like(normalized_distance) * 0.6 + 0.4  # [0.4, 1.0]

    # Compute heuristic scores with diverse nonlinear transformations
    score_distance = torch.exp(-normalized_distance) * rand_weights  # Favor shorter distances
    score_capacity = (current_load.unsqueeze(1) - delivery_node_demands) / current_load.unsqueeze(1)  # Favor feasible deliveries
    score_time_window = torch.sigmoid((arrival_times - time_windows[:, 0].unsqueeze(0)).clamp(min=0))  # Favor adherence to time windows
    score_length = (current_length.unsqueeze(1) - normalized_distance) / current_length.unsqueeze(1)  # Favor feasible routing length

    # Combine heuristic scores
    heuristic_scores = score_distance + score_capacity + score_time_window + score_length

    # Incorporate a randomness factor to diversify the search
    heuristic_scores += torch.rand_like(heuristic_scores) * 0.05

    # Ensure scores are normalized and bounded
    heuristic_scores = torch.clamp(heuristic_scores, min=0, max=1)

    return heuristic_scores