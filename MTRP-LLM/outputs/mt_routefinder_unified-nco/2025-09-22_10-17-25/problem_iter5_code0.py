import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v4
    
    # Adjust the calculation of distance score based on a logarithmic method
    distance_heuristic = 1 / (torch.log(current_distance_matrix + 1) + 1)  # Modified distance heuristic with logarithmic
    
    # Change how the delivery score is calculated
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.3
    
    # Modify the calculation of load score
    load_score = current_load * 0.5 + current_load.max() * 0.1
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.3

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores