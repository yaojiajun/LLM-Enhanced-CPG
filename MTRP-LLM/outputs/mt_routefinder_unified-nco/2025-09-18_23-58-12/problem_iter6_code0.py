import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Enhanced randomization and adaptive weights with weighted distance metrics
    random_scores = torch.randn_like(current_distance_matrix) * 0.2
    weight_factor = 0.8  # Weight for distance-based scores
    distance_based_scores = current_distance_matrix * torch.rand_like(current_distance_matrix) * weight_factor
    
    # Additional heuristic manipulation or calculations
    # Add your custom heuristic operations here
    
    return distance_based_scores + random_scores