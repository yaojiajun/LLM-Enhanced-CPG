import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Add randomness to the heuristic score matrix
    random_scores = torch.rand_like(current_distance_matrix) * 2 - 1  # Random scores between -1 and 1
    return F.softmax(current_distance_matrix + random_scores, dim=1)