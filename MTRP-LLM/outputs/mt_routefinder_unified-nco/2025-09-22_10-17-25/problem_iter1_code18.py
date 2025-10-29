import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Change in the calculation for 'current_distance_matrix' heuristic score
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.6  # Enhanced randomness with a different scaling factor
    
    # Change in the calculation for 'delivery_node_demands' heuristic score
    demand_scores = (current_load_open.unsqueeze(1) - delivery_node_demands) * 0.7 + torch.max(
        current_load_open) / 2 + torch.randn_like(current_distance_matrix) * 0.4 # Update based on load efficiency and randomness

    # Same calculation for 'current_load' heuristic score
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine all heuristic scores
    overall_scores = normalized_distance_scores + demand_scores + enhanced_noise

    return overall_scores