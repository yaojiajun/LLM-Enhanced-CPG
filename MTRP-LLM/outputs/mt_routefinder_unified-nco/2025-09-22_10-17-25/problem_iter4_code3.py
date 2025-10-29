import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modified heuristics with specific changes in distance, delivery, and load calculations
    
    # Adjust the calculation of distance score based on a different transformation
    distance_heuristic = 2 / (current_distance_matrix + 1e-5)  # Updated distance heuristic
    
    # Change the delivery score calculation by incorporating a varying factor
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) * (current_load.sum() / (current_load.sum() + 1)) * 0.8  # Modified delivery heuristic
    
    # Modify the load score calculation using a different approach
    load_score = (current_load * 0.3 + current_load.max() * 0.2) / (current_load + 1)  # Updated load heuristic
    
    # Generate random noise matrix for additional exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine the heuristic scores with the updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores