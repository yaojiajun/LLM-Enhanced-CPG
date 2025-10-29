import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified version of the heuristics function with changes only in handling current_distance_matrix, delivery_node_demands, and current_load
    
    # Adjusted distance-based heuristic score by reweighting and adding noise
    distance_heuristic = (-current_distance_matrix / torch.max(current_distance_matrix)) + torch.randn_like(current_distance_matrix) * 0.5
    
    # Modified delivery score to emphasize critical demand nodes with load balancing consideration
    delivery_score = (2 * delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.7 + torch.max(delivery_node_demands) * 0.3 + torch.randn_like(current_distance_matrix) * 0.4
    
    # Integrated pickup score with a dynamic load balancing approach and additional noise
    pickup_score = (pickup_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(pickup_node_demands) * 0.2 + torch.randn_like(current_distance_matrix) * 0.3
    
    # Combine the modified heuristic scores with noise for exploration
    total_scores = distance_heuristic + delivery_score + pickup_score + torch.randn_like(current_distance_matrix) * 0.3

    return total_scores