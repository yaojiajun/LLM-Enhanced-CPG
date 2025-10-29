import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Mutated version incorporating enhanced randomness and adaptability
    heuristics_scores = torch.randint(-100, 100, size=current_distance_matrix.size())  # Example of incorporating enhanced randomness
    
    return heuristics_scores