import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the calculation for normalized_distance_scores to increase emphasis on distance and randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.9

    # Modify the demand_scores calculation to give higher weight to delivery demands and randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.6

    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Remainder of the function remains unchanged from the original implementation

    return cvrp_scores