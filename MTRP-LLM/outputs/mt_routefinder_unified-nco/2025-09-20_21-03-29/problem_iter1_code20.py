import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify how the heuristic score matrix is computed for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    
    # Example modification:
    distance_heuristic = torch.exp(-current_distance_matrix)  # Modify how distance affects the heuristic score
    delivery_score = delivery_node_demands / torch.maximum(current_load, torch.tensor(1e-8))  # Modify delivery score calculation
    pickup_score = torch.maximum(pickup_node_demands - current_load, torch.tensor(0.0))  # Modify pickup score calculation
    
    total_score = distance_heuristic + delivery_score - pickup_score
    
    return total_score