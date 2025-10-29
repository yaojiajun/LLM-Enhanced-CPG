import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    epsilon = 1e-8
    random_scores = torch.rand_like(current_distance_matrix)

    # Incorporating insights from prior heuristics
    scaled_dist = current_distance_matrix / (current_length.unsqueeze(1) + epsilon)
    load_ratio = current_load.unsqueeze(1) / (delivery_node_demands + epsilon)
    open_load_ratio = current_load_open.unsqueeze(1) / (delivery_node_demands_open + epsilon)
    time_diff = arrival_times[:, :-1] - arrival_times[:, 1:]
    time_diff = torch.cat([time_diff, torch.zeros_like(arrival_times[:, -1:])], dim=1)
    time_ratio = time_diff / (time_windows[:, 1] - arrival_times + epsilon)

    heuristic_scores = (scaled_dist + load_ratio + open_load_ratio + time_ratio + random_scores) / 5.0

    return torch.clamp(heuristic_scores, min=-1e6, max=1e6)