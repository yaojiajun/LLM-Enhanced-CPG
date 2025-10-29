import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modified heuristics calculation for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    
    # New distance heuristic calculation with added randomness and inverse scaling
    distance_heuristic = -1.0 / (current_distance_matrix + torch.randn_like(current_distance_matrix) * 0.5)
    
    # Adjust delivery score calculation with increased sensitivity to load
    delivery_score = (delivery_node_demands * current_load).unsqueeze(1) * 0.5
    
    # Refine pickup score based on the reciprocals of pickups and load
    pickup_score = 1.0 / (pickup_node_demands * current_load).clamp(min=1e-6)
    
    # Combine the modified heuristics with the existing components for the overall scores
    overall_scores = distance_heuristic + delivery_score + pickup_score

    return overall_scores