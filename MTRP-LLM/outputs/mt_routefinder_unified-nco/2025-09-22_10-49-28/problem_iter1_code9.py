import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the computation of the normalized distance-based heuristic score with added diversity through randomness
    distance_heuristic = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Modify the demand-based heuristic score computation with emphasis on high-demand nodes and enhanced randomness
    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the modified heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = distance_heuristic + delivery_score + enhanced_noise

    # Keep the rest of the function unchanged as per original design

    return cvrp_scores