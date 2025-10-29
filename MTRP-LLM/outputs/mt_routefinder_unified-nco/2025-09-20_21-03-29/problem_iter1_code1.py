import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the computation for current_distance_matrix, delivery_node_demands, and current_load
    distance_heuristic = torch.exp(-current_distance_matrix)
    delivery_score = 1 / (delivery_node_demands + 1e-8)
    pickup_score = 1 / (pickup_node_demands + 1e-8)
    
    # Combine the heuristics into total score matrix
    total_score = distance_heuristic + delivery_score + pickup_score
    
    return total_score