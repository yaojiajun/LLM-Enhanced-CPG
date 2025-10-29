import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the calculation of the heuristic score matrix based on current_distance_matrix, delivery_node_demands, and current_load
    distance_heuristic = 1 / (current_distance_matrix + 1e-8)  # Add epsilon for numerical stability
    delivery_score = torch.div(delivery_node_demands, current_load + 1e-8)  # Avoid division by zero with epsilon
    pickup_score = pickup_node_demands / (current_load + 1e-8)  # Add epsilon for stability

    # Integrate distance_heuristic, delivery_score, and pickup_score into the total score matrix
    total_score = distance_heuristic - delivery_score[:, None] + pickup_score[:, None]

    return total_score