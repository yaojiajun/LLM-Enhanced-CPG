import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5 - Modified Version
    
    # Adjust the calculation of distance score based on a modified normalization approach
    distance_heuristic = torch.exp(-torch.sqrt(current_distance_matrix)) / (torch.max(current_distance_matrix) + 0.5)  # Modified normalized distance heuristic
    
    # Change how the delivery score is calculated by considering a penalty scheme for high demands
    delivery_score = (delivery_node_demands - torch.min(delivery_node_demands)) * ((delivery_node_demands > 0).float()) * (-0.3)
    
    # Adapt the load score calculation by penalizing low load values
    load_score = (current_load.mean() - current_load.min()) - (current_load - current_load.mean()).abs() * 0.7
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores