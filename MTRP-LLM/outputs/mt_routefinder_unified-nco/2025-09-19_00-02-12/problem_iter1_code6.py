import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Calculate load feasibility indicator
    load_feasibility = torch.where(current_load.unsqueeze(1) >= delivery_node_demands, torch.ones_like(delivery_node_demands), torch.zeros_like(delivery_node_demands))
    
    # Calculate open load feasibility indicator
    open_load_feasibility = torch.where(current_load_open.unsqueeze(1) >= delivery_node_demands_open, torch.ones_like(delivery_node_demands_open), torch.zeros_like(delivery_node_demands_open))
    
    # Calculate time window feasibility indicator
    time_window_feasibility = torch.where((arrival_times >= time_windows[:, 0].unsqueeze(0)) & (arrival_times <= time_windows[:, 1].unsqueeze(0)), torch.ones_like(arrival_times), torch.zeros_like(arrival_times))
    
    # Calculate length feasibility indicator
    length_feasibility = torch.where(current_length.unsqueeze(1) >= current_distance_matrix, torch.ones_like(current_distance_matrix), torch.zeros_like(current_distance_matrix))
    
    # Combine feasibility indicators into a heuristic score matrix
    heuristic_scores = load_feasibility + open_load_feasibility + time_window_feasibility + length_feasibility
    
    return heuristic_scores