import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5 (mutated version)
    
    # Adjust the calculation of distance score based on a new customization approach
    distance_heuristic = torch.exp(-torch.sqrt(current_distance_matrix)) / (torch.max(current_distance_matrix) + 1e-6) * 0.8  # Updated normalized distance heuristic
    
    # Change how the delivery score is calculated by adding a penalty for very high demands
    delivery_score = (torch.min(delivery_node_demands) - delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.4
    
    # Modify the calculation of load score by incorporating a penalty for slight load imbalances
    load_score = (current_load.max() - current_load.min()) - (current_load - current_load.mean()).abs() * 0.4
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores