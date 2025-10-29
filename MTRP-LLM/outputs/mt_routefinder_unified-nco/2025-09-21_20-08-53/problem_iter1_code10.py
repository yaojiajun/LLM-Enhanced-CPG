import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # New calculation for distance-based heuristic score with modified emphasis on distance
    distance_heuristic = -(current_distance_matrix ** 2) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # New calculation for demand-based heuristic score with adjusted sensitivity to remaining capacity
    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    total_scores = distance_heuristic + delivery_score + enhanced_noise

    # Return the final heuristic score matrix
    return total_scores