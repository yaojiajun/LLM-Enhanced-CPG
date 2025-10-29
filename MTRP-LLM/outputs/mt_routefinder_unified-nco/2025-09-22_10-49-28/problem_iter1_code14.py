import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores_v2
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6

    demand_scores_v2 = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.5 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    enhanced_noise_v2 = torch.randn_like(current_distance_matrix) * 1.5

    cvrp_scores_v2 = normalized_distance_scores_v2 + demand_scores_v2 + enhanced_noise_v2

    # Keep other parts of the function unchanged

    return cvrp_scores_v2