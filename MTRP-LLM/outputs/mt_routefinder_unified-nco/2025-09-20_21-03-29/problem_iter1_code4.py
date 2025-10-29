import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Modify the calculations related to current_distance_matrix, delivery_node_demands, and current_load here
    distance_heuristic = 1 / (current_distance_matrix + 1e-8)  # Ensure numerical stability
    delivery_score = delivery_node_demands / (current_load + 1e-8)  # Ensure numerical stability
    pickup_score = pickup_node_demands / (current_load + 1e-8)  # Ensure numerical stability
    
    total_score = distance_heuristic + delivery_score - pickup_score
    
    return total_score