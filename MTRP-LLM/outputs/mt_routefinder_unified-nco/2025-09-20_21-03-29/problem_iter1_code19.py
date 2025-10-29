import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify heuristics_v1 by incorporating insights on distance, delivery demands, and current load
    distance_factor = 1.0 / (current_distance_matrix + 1e-8)  # Ensure numerical stability
    delivery_score = delivery_node_demands / (current_load + 1e-8)  # Delivery demand vs. remaining load
    pickup_score = pickup_node_demands / (current_load + 1e-8)  # Pickup demand vs. remaining load
    
    total_score = distance_factor + delivery_score - pickup_score  # Combined heuristic score
    
    return total_score