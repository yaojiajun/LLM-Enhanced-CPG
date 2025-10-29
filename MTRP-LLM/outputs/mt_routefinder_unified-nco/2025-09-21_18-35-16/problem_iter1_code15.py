import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the distance heuristic calculation for increased emphasis
    modified_distance_scores = -current_distance_matrix / (torch.max(current_distance_matrix) + 1e-8) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Re-calculate delivery-based scores with variance and augmented factor
    delivery_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.4

    # Maintain the previous noise enhancement factor
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine modified distance and delivery scores with noise for diversified strategy
    heuristic_scores = modified_distance_scores + delivery_scores + enhanced_noise

    # Include other heuristics calculations as before for consistency
    # ...

    return heuristic_scores