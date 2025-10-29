import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v3
    
    # Adjust the calculation of distance score based on a squared method
    distance_heuristic = current_distance_matrix ** 2  # Squared distance heuristic
    
    # Change how the delivery score is calculated
    delivery_score = delivery_node_demands * 1.5 - current_load.max() * 0.7
    
    # Modify the calculation of load score
    load_score = current_load * 0.6 + current_load.max() * 0.3
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.6

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores