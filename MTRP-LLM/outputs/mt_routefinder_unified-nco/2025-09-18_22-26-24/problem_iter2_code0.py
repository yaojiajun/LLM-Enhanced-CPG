import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Your improved heuristic implementation here
    shuffled_distance_matrix = current_distance_matrix.clone()
    
    # Introduce randomness based on node characteristics
    node_characteristics = torch.cat((delivery_node_demands.unsqueeze(1), time_windows[:, 1].unsqueeze(1)), dim=1)
    shuffled_indices = torch.randperm(shuffled_distance_matrix.shape[1])
    shuffled_distance_matrix = torch.index_select(shuffled_distance_matrix, dim=1, index=shuffled_indices)
    
    return shuffled_distance_matrix * torch.randn_like(shuffled_distance_matrix)