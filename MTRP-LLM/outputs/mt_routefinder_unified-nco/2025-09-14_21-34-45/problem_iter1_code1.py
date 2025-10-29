import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Compute heuristic score based on various factors such as distance, load constraints, time windows, and duration limits
    distance_score = -current_distance_matrix # Higher distance means lower score
    
    load_constraint_score = torch.where(current_load.unsqueeze(2) < delivery_node_demands.unsqueeze(0), torch.tensor(-10.0), torch.tensor(0.0))
    
    load_open_constraint_score = torch.where(current_load_open.unsqueeze(2) < delivery_node_demands_open.unsqueeze(0), torch.tensor(-10.0), torch.tensor(0.0))
    
    time_window_score = torch.where((arrival_times < time_windows[:, 0].unsqueeze(0)) | (arrival_times > time_windows[:, 1].unsqueeze(0)), torch.tensor(-5.0), torch.tensor(0.0))
    
    duration_constraint_score = torch.where(current_length.unsqueeze(2) < current_distance_matrix, torch.tensor(-1.0), torch.tensor(0.0))
    
    # Combine all scores
    total_score = distance_score + load_constraint_score + load_open_constraint_score + time_window_score + duration_constraint_score
    
    return total_score