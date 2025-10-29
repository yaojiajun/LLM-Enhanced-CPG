import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Calculate the ratio of remaining load to delivery demand for each node
    load_ratio = current_load.unsqueeze(1) / delivery_node_demands.unsqueeze(0)

    # Calculate the ratio of remaining load to pickup demand for each node
    load_ratio_open = current_load_open.unsqueeze(1) / pickup_node_demands.unsqueeze(0)

    # Calculate the normalized distance matrix
    norm_distance_matrix = F.normalize(current_distance_matrix, p=2, dim=1)

    # Combine different ratios and normalized distances to compute heuristic scores
    heuristic_scores = (load_ratio - load_ratio_open) + 0.5 * norm_distance_matrix

    return heuristic_scores