import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v4
    
    # Adjust the calculation of distance score based on a square root transformation
    distance_heuristic = torch.sqrt(current_distance_matrix) / (torch.max(current_distance_matrix) + 1e-6)  # Modified distance heuristic with square root
    
    # Modify the delivery score calculation by considering the square of demand differences
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) ** 2 * ((delivery_node_demands > 0).float()) * 0.5
    
    # Change how the load score is calculated by leveraging the reciprocal of current load
    load_score = 1 / (current_load + 1e-6) * 0.4  + current_load.max() * 0.2
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.3

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores