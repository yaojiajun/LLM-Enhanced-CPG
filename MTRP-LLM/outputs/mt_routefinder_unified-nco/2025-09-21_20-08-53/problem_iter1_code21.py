import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    # Modify the distance-based heuristic score matrix with added randomness and diversity
    normalized_distance_scores_v2 = -current_distance_matrix / torch.std(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    # Adjust the demand-based heuristic score matrix for high-demand nodes with increased emphasis and randomness
    demand_scores_v2 = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce enhanced noise for exploration with elevated diversity
    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores with diverse strategies for balanced exploration
    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    # Keep the code structure, function signature, and all other parts unchanged except the modifications mentioned above

    return cvrp_scores_v2