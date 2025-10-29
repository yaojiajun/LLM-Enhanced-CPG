import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5
    
    # Adjust the calculation of distance score based on a new power transformation
    distance_heuristic = torch.exp(-torch.pow(current_distance_matrix, 0.5)) / (torch.max(current_distance_matrix) + 1e-6)  # Power-transformed distance heuristic
    
    # Change how the delivery score is calculated with a different scaling approach
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.2
    
    # Modify the calculation of load score by incorporating a sinusoidal penalty function
    load_score = (torch.sin(current_load) + 1) * 0.4
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores