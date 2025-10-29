import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Customized heuristics for model v5
    
    # Adjust the calculation of distance score by taking the reciprocal of the distance matrix and adding a small constant
    distance_heuristic = 1 / (current_distance_matrix + 1e-6)  # Reciprocal of distance heuristic
    
    # Modify the delivery score calculation by multiplying the demand with a non-linear transformation
    delivery_score = torch.exp(-delivery_node_demands)  # Exponential transformation of delivery demand
    
    # Change how the load score is calculated by considering a different function of the current load
    load_score = torch.log(current_load + 1)  # Logarithmic transformation of current load
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.3

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores