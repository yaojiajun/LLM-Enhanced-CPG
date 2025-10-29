import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Improvements made for the heuristics function v2

    # Adjust the calculation of distance score based on a different method
    distance_heuristic = torch.exp(-torch.square(current_distance_matrix)) * 0.9  # Updated distance heuristic
    
    # Change how the delivery score is calculated
    delivery_score = (torch.max(delivery_node_demands) - delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.4
    
    # Modify the calculation of load score
    load_score = (current_load * 1.5 - current_load.mean()) * 0.4
    
    # Compute a random matrix for diverse exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.5

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores