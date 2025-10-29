import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Mutated heuristics for model v4
    
    # Adjust the calculation of distance score based on a sigmoid method
    distance_heuristic = torch.sigmoid(-current_distance_matrix) / (torch.max(current_distance_matrix) + 1e-6)  # Modified distance heuristic with sigmoid
    
    # Change how the delivery score is calculated
    delivery_score = (torch.max(delivery_node_demands) / delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.8
    
    # Modify the calculation of load score
    load_score = current_load * 0.6  + current_load.max() * 0.3
    
    # Compute a random matrix for added diversity
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.3

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores