import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize parameters to prevent inf or -inf
    eps = 1e-8
    delivery_node_demands_norm = delivery_node_demands + eps
    current_load_norm = current_load + eps
    delivery_node_demands_open_norm = delivery_node_demands_open + eps
    current_load_open_norm = current_load_open + eps
    
    # Heuristic score computation
    score = torch.zeros(current_distance_matrix.size())
    
    # Add randomness to scoring
    random_factor = torch.rand(score.size())
    
    # Compute heuristic scores with controlled randomness
    score = current_distance_matrix / (delivery_node_demands_norm.unsqueeze(0) * current_load_norm.unsqueeze(1) * delivery_node_demands_open_norm.unsqueeze(0) * current_load_open_norm.unsqueeze(1)) * random_factor
    
    # Mask invalid values
    score[torch.isinf(score)] = 0
    score[torch.isnan(score)] = 0
    
    return score