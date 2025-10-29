import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v6
    
    # Adjust the calculation of distance score by taking a log transformation
    distance_heuristic = -torch.log(current_distance_matrix + 1e-6) / (torch.max(current_distance_matrix) + 1e-6)  # Log distance heuristic
    
    # Change how the delivery score is calculated by normalizing demand values
    delivery_score = (delivery_node_demands - delivery_node_demands.mean()) / (delivery_node_demands.std() + 1e-6) * 0.2
    
    # Modify the calculation of load score by considering a penalty for exceeding a certain threshold
    load_threshold = current_load.mean() + current_load.std()
    load_score = torch.where(current_load > load_threshold, current_load - load_threshold, torch.tensor(0.0))
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores