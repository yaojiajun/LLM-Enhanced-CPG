import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Calculate a modified version of the heuristic score matrix based on current_distance_matrix, delivery_node_demands, and current_load
    distance_heuristic = torch.exp(-current_distance_matrix)  # Example transformation of distance heuristic
    delivery_score = torch.log(delivery_node_demands + 1)  # Example transformation of delivery score
    load_ratio = current_load / (delivery_node_demands + 1e-8)  # Avoiding division by zero with a small epsilon
    load_score = 1 / (1 + torch.exp(-load_ratio))  # Example transformation of load score

    total_score = distance_heuristic + delivery_score + load_score

    return total_score