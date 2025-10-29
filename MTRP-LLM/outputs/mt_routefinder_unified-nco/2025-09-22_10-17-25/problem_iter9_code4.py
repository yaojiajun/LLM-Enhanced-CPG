import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5 - Updated version
    
    # Modify distance heuristic calculation to include squared inverse of distances
    distance_heuristic = 1 / (current_distance_matrix**2 + 1e-6)  # Updated distance heuristic
    
    # Adjust delivery score calculation by adding a penalty for high demands
    delivery_score = (delivery_node_demands - torch.mean(delivery_node_demands)) * ((delivery_node_demands > 0).float()) * 0.2
    
    # Update load score calculation to consider the ratio of current load to maximum load
    load_score = (current_load / (current_load.max() + 1e-6)) * 0.4
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores