import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Compute heuristic scores based on a combination of domain-specific insights and enhanced randomness
    distance_scores = torch.rand_like(current_distance_matrix)  # Random distance scores
    demand_scores = 1 / (1 + delivery_node_demands)  # Inverse of delivery demands as scores
    time_scores = 1 / (1 + arrival_times.sum(dim=0))  # Inverse of total arrival times as scores
    pickup_scores = torch.where(pickup_node_demands > 0, -pickup_node_demands, torch.zeros_like(pickup_node_demands))  # Negative scores for pickups
    
    heuristic_scores = distance_scores + demand_scores - time_scores + pickup_scores
    return heuristic_scores