import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Updated distance heuristic calculation
    # Compute the distance-based heuristic score matrix with random noise
    distance_scores = -current_distance_matrix / torch.mean(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7  # Increased randomness factor
    
    # Modified delivery score calculation
    # Compute the demand-based heuristic score matrix with a higher reward for satisfying delivery and reduced noise
    delivery_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 1.3 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.2  # Fine-tuned noise

    # Adjusted load score calculation
    # Compute the load-based heuristic score matrix with emphasis on ensuring load capacity is not exceeded
    load_scores = (current_load_open.unsqueeze(1) - torch.min(current_load_open)).abs() * 0.8 + torch.randn_like(
        current_distance_matrix) * 0.4  # More focus on load balancing

    # Combine the updated heuristic scores
    cvrp_scores = distance_scores + delivery_scores + load_scores

    return cvrp_scores