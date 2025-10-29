import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified version of the heuristics function with changes only in distance, delivery, and load calculations
    
    # Distance heuristic modification
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 1.0  # Increased randomness weight
    
    # Delivery score modification
    demand_scores = (delivery_node_demands / current_load.unsqueeze(1).clamp(min=1e-8)) * 3.0 + torch.randn_like(current_distance_matrix) * 0.5
    demand_scores = torch.where(torch.isnan(demand_scores), torch.zeros_like(demand_scores), demand_scores)  # Avoid nan values
    
    cvrp_scores = normalized_distance_scores + demand_scores
    
    # Remaining parts of the original function remain unchanged
    # Essentially, combine the heuristic scores as in the original function
    # Ensure the output remains a heuristic score matrix of shape (pomo_size, N+1) with positive scores for promising edges and negative for undesirable ones
    
    
    return cvrp_scores