import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5
    
    # Adjust the calculation of distance score based on a new approach
    distance_heuristic = torch.exp(-torch.sqrt(current_distance_matrix)) / (torch.max(current_distance_matrix) + 1e-6)  # Normalized distance heuristic
    
    # Modify the delivery score calculation to emphasize higher demands
    delivery_score = torch.log(delivery_node_demands + 1) * 0.2
    
    # Change how load score is calculated by introducing penalties for lower and upper load limits
    load_score = torch.abs(current_load - 0.5*(current_load.min() + current_load.max())) - torch.sin(current_load) * 0.3
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores