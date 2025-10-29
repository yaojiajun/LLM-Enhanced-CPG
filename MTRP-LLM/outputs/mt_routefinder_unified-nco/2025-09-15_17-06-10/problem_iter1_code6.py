import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Some example heuristic computations
    score1 = torch.sigmoid(current_distance_matrix)
    score2 = F.relu(torch.sin(current_distance_matrix))
    heuristic_scores = score1 * score2

    return heuristic_scores