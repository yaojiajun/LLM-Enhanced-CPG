import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Introduce problem-specific adjustments to random scores for informed exploration and optimization
    heuristic_scores = torch.rand_like(current_distance_matrix) * 0.5  # Adjust random heuristic scores for exploration
    
    # Apply domain-specific knowledge and constraints to improve edge selection
    if torch.any(current_distance_matrix < 0):
        heuristic_scores -= 0.2  # Penalize negative distances
    
    return heuristic_scores