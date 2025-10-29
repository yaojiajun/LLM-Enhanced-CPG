import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5 - Modified Version
    
    # Adjust the calculation of distance score based on a new transformation
    distance_heuristic = torch.exp(-torch.sqrt(current_distance_matrix/10)) / (torch.max(current_distance_matrix) + 1e-6)  # Modified distance heuristic
    
    # Change how the delivery score is computed by penalizing nodes with high demands
    delivery_score = (-delivery_node_demands) * ((delivery_node_demands > 0).float()) * 0.4
    
    # Modify load score calculation by emphasizing load imbalance penalties
    load_score = ((current_load.max() - current_load.mean()) - (current_load - current_load.mean()).abs()) * 0.6
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores