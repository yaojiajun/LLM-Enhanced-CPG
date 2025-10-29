import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristic calculations for current_distance_matrix, delivery_node_demands, and current_load
    modified_distance_scores = torch.exp(-current_distance_matrix)  # Incorporate exponential transformation for distance heuristic
    modified_delivery_scores = (current_load / (delivery_node_demands + 1e-8))  # Adjust delivery score computation
    modified_pickup_scores = torch.log(torch.sqrt(delivery_node_demands * pickup_node_demands + 1)) / (current_load + 1e-8)  # Revise pickup score calculation
    
    # Combine the modified heuristics with existing scores
    overall_scores = modified_distance_scores + modified_delivery_scores + modified_pickup_scores
    
    return overall_scores