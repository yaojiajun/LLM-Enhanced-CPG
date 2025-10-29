import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Customized heuristics for model v5
    
    # Adjust the calculation of distance score based on a new transformation approach
    distance_heuristic = torch.exp(-torch.square(current_distance_matrix)) / (torch.max(current_distance_matrix) + 1e-6)  # Transformed distance heuristic
    
    # Change delivery score calculation to emphasize importance of demand satisfaction
    delivery_score = torch.sqrt(torch.abs(torch.min(delivery_node_demands) - delivery_node_demands)) * ((delivery_node_demands > 0).float()) * 0.3
    
    # Modify load score calculation to focus on balancing load distribution
    load_score = (current_load.min() - current_load) - (current_load - current_load.mean()).abs() * 0.5
    
    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores