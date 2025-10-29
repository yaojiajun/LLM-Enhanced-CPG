import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristics calculations specifically for current_distance_matrix, delivery_node_demands, and current_load

    # Compute a distance-based heuristic score matrix with a different approach
    distance_heuristic = torch.exp(-current_distance_matrix / torch.std(current_distance_matrix)) * 0.5

    # Compute a delivery node demand-based heuristic score matrix with adjusted weights
    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3

    # Introduce controlled randomness for diversification
    random_noise = torch.randn_like(current_distance_matrix) * 0.5

    # Combine the modified heuristic scores with diversified strategies
    total_scores = distance_heuristic + delivery_score + random_noise

    # Return the final heuristic score matrix
    return total_scores