import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Compute heuristic indicators
    distance_heuristic = 1.0 / (current_distance_matrix + 1e-8)  # Add small epsilon for numerical stability
    delivery_score = 2.0 / (delivery_node_demands + 1e-8)  # Add small epsilon for numerical stability
    pickup_score = 1.0 / (pickup_node_demands + 1e-8)  # Add small epsilon for numerical stability
    
    # Introduce controlled randomness
    distance_heuristic += torch.randn_like(distance_heuristic) * 0.1
    delivery_score += torch.randn_like(delivery_score) * 0.1
    pickup_score += torch.randn_like(pickup_score) * 0.1
    
    # Combine heuristic indicators
    total_score = distance_heuristic + delivery_score + pickup_score
    
    return total_score