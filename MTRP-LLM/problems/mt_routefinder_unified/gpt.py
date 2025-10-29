import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Initialize score matrix
    score_matrix = torch.zeros_like(current_distance_matrix)
    
    # Assess infeasibility of nodes through demand
    delivery_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * (delivery_node_demands > 0).float()
    
    # Enhanced distance scoring with adaptive penalties and scaling
    scaled_distance_matrix = current_distance_matrix / (1 + delivery_node_demands.unsqueeze(0).clamp(max=10.0))  # Adjust distance based on demand
    penalty_factor = (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1e-5)).clamp(max=5.0)
    adjusted_distance_matrix = scaled_distance_matrix * (1 + penalty_factor)  # More sensitive to demand
    base_delivery_score = -adjusted_distance_matrix * delivery_feasibility
    
    # Enhanced penalties and considerations for open deliveries
    feasible_open_delivery = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float() * (delivery_node_demands_open > 0).float()
    
    # Introduced interaction between pickups and deliveries with controlled penalties for open routes
    open_route_weights = torch.sigmoid(current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)) * \
                         torch.relu((time_windows[:, 1].unsqueeze(0) - arrival_times) / (time_windows[:, 1] - time_windows[:, 0] + 1e-5).unsqueeze(0))
    opened_delivery_penalty = torch.where(feasible_open_delivery > 0, 
                                          -0.7 * current_distance_matrix * open_route_weights + 1.2 / (1 + torch.exp(current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0))), 
                                          -1e10 * torch.ones_like(current_distance_matrix).to(current_distance_matrix.device))
    
    # Combine scores considering both delivery and open inspections, avoiding extremes
    combined_delivery_score = base_delivery_score + opened_delivery_penalty
    combined_delivery_score[combined_delivery_score == -1e10] = 0  # avoid extreme penalties in overlap
    
    # Adjust randomness based on score
    variability_factor = torch.rand_like(combined_delivery_score) * (3.0 - torch.sigmoid(combined_delivery_score))  # Increased randomness variability
    final_scores = combined_delivery_score + variability_factor
    
    # Normalize to avoid extremes
    min_score = final_scores.min()
    max_score = final_scores.max()
    normalized_scores = (final_scores - min_score) / (max_score - min_score + 1e-5)  # to prevent division by zero
    
    # Assign to score matrix
    score_matrix += normalized_scores

    return score_matrix
