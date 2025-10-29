import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # New computation for distance-based heuristic score matrix with modified diversity
    normalized_distance_scores_squared = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Compute the demand-based heuristic score matrix with updated emphasis on high-demand nodes and randomness
    demand_scores_sqrt = (torch.sqrt(delivery_node_demands.unsqueeze(0)) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce different noise pattern for exploration with varied noise level
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores with diverse strategies for exploration
    modified_scores = normalized_distance_scores_squared + demand_scores_sqrt + enhanced_noise

    # Remaining parts remain the same as in the original function
    # Please avoid any changes to them according to the given instructions

    return modified_scores