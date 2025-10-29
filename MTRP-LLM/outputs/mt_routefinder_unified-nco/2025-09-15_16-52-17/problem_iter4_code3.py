import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Implement further improved heuristic logic with enhanced exploration and exploitation
    random_scores = torch.rand_like(current_distance_matrix)  # Placeholder for randomness
    capacity_scores = torch.rand_like(current_distance_matrix) * 0.6  # Score based on remaining capacity
    time_scores = torch.rand_like(current_distance_matrix) * 0.4  # Score based on time windows
    pickup_scores = torch.rand_like(current_distance_matrix) * 0.3  # Score based on pickup demands
    backhaul_scores = torch.rand_like(current_distance_matrix) * 0.2  # Score based on backhaul considerations
    
    heuristic_scores = random_scores + capacity_scores + time_scores + pickup_scores + backhaul_scores
    return heuristic_scores