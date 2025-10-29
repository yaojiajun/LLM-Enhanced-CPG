import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize distance matrix
    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / max_distance if max_distance > 0 else current_distance_matrix

    # Calculate load and demand feasibility
    feasible_load = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Time window feasibility
    arrival_within_window = ((arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times > time_windows[:, 0].unsqueeze(0))).float()

    # Dynamic weights based on feasibility
    rand_weights = torch.rand_like(normalized_distance) * 0.5 + 0.5  # [0.5, 1.0]
    distance_weight = 1 - rand_weights
    feasibility_weight = rand_weights

    # Compute heuristic scores
    score_distance = (1 - normalized_distance) * distance_weight  # Favor shorter distances
    score_load = feasible_load * feasibility_weight  # High score for feasible deliveries
    score_time = arrival_within_window * feasibility_weight  # Favor routes meeting time window constraints

    # Combine heuristic scores with adjusted scaling
    heuristic_scores = score_distance + score_load + score_time

    # Ensure scores are normalized and bounded
    heuristic_scores = torch.clamp(heuristic_scores, min=0, max=1)

    return heuristic_scores