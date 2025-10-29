import torch
import numpy as np
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize input data
    norm_distance = F.normalize(current_distance_matrix, p=2, dim=1)
    norm_load = F.normalize(current_load.view(-1, 1), p=2, dim=0)
    norm_load_open = F.normalize(current_load_open.view(-1, 1), p=2, dim=0)
    norm_length = F.normalize(current_length.view(-1, 1), p=2, dim=0)

    # Compute heuristic scores using diverse activation functions and combined indicators
    score1 = torch.sigmoid(norm_distance)
    score2 = F.relu(torch.sin(norm_load) * torch.cos(norm_load_open))
    score3 = torch.tanh(norm_length)
    
    heuristic_scores = score1 * score2 * score3

    return heuristic_scores