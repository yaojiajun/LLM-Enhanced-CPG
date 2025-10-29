import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    edge_scores = torch.rand(current_distance_matrix.size())  # Example of random assignment, replace with actual heuristic computation
    edge_scores = F.normalize(edge_scores, dim=1)  # Normalize scores along axis 1 for better comparison
    return edge_scores