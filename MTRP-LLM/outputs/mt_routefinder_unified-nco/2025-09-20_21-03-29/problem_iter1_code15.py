import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Compute modified distance heuristic
    distance_heuristic = F.leaky_relu(1.0 / (current_distance_matrix + 1e-8), negative_slope=0.2)
    
    # Compute modified delivery score
    delivery_score = F.sigmoid(delivery_node_demands / (current_load + 1e-8))
    
    # Compute modified pickup score
    pickup_score = F.hardtanh(pickup_node_demands / (current_load + 1e-8), min_val=0.1, max_val=0.9)
    
    # Combine heuristic scores
    total_score = distance_heuristic + delivery_score - pickup_score
    
    return total_score