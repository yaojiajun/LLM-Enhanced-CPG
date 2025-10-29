import torch
import numpy as np
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Normalize inputs
    norm_distance_matrix = torch.nn.functional.normalize(current_distance_matrix, p=2, dim=1)
    norm_delivery_node_demands = torch.nn.functional.normalize(delivery_node_demands, p=2, dim=0)
    norm_current_load = torch.nn.functional.normalize(current_load, p=2, dim=0)
    norm_delivery_node_demands_open = torch.nn.functional.normalize(delivery_node_demands_open, p=2, dim=0)
    norm_current_load_open = torch.nn.functional.normalize(current_load_open, p=2, dim=0)
    norm_time_windows = torch.nn.functional.normalize(time_windows, p=2, dim=0)
    norm_arrival_times = torch.nn.functional.normalize(arrival_times, p=2, dim=1)
    norm_pickup_node_demands = torch.nn.functional.normalize(pickup_node_demands, p=2, dim=0)
    norm_current_length = torch.nn.functional.normalize(current_length, p=2, dim=0)
    
    # Combine multiple indicators with weighted scoring
    score1 = torch.sigmoid(norm_distance_matrix) * 0.5
    score2 = F.relu(torch.sqrt(norm_delivery_node_demands)) * 0.3
    score3 = torch.tanh(norm_current_load) * 0.2
    
    heuristic_scores = score1 + score2 + score3

    return heuristic_scores