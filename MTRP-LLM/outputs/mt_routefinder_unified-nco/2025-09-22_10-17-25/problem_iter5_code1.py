import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v4
    
    # Adjust the calculation of distance score based on a squared exponential method
    distance_heuristic = torch.exp(-(current_distance_matrix**2)) / (torch.max(current_distance_matrix) + 1e-6)  # Modified distance heuristic with squared exponential
    
    # Change how the delivery score is calculated by considering a penalty for high demands
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.2
    
    # Modify the calculation of load score by emphasizing even load distribution
    load_score = (current_load - current_load.mean()).abs() * 0.6
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.3

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores