import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the heuristics calculations related to current_distance_matrix, delivery_node_demands, and current_load
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7  # Example modification, keeping randomness

    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5  # Example modification, keeping randomness

    # Placeholder operations for other parts of the original code without modification
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0
    
    criticality_weights = torch.where(torch.clamp(arrival_times - time_windows[:, 1].unsqueeze(0), min=0) > 0, 1.7, 0.3)
    
    total_scores = normalized_distance_scores + demand_scores + enhanced_noise

    return total_scores