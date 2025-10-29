import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Modify the calculation for the heuristic score based on current_distance_matrix, delivery_node_demands, and current_load
    distance_heuristic = torch.exp(-current_distance_matrix)  # Example modification

    delivery_score = torch.log(1 + delivery_node_demands)  # Example modification

    load_factor = current_load / (delivery_node_demands + 1e-8)  # Example modification

    total_score = distance_heuristic + delivery_score + load_factor + torch.randn_like(current_distance_matrix)  # Incorporating randomness

    return total_score