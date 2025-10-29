import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Capacity and Time Constraints
    capacity_constraints = (current_load[:, None] >= delivery_node_demands[None, :])
    time_constraints = (arrival_times < time_windows[:, 1][None, :]) & (arrival_times > time_windows[:, 0][None, :])

    # Calculate base scores from distance matrix, inversely proportional to distance
    base_scores = 1 / (current_distance_matrix + 1e-6)  # Avoid division by zero
    heuristic_scores += base_scores

    # Penalize for capacity violations (negative scores)
    heuristic_scores[~capacity_constraints] -= 1e3

    # Penalize for time window violations (negative scores)
    heuristic_scores[~time_constraints] -= 1e3

    # Adding pickup demands consideration
    pickup_capacity_constraints = (current_load[:, None] + pickup_node_demands[None, :] <= delivery_node_demands_open[None, :])
    heuristic_scores[pickup_capacity_constraints] += 0.5  # Favor pickup candidates that fit capacity

    # Random noise to maintain exploration and escape local optima
    noise = torch.rand_like(heuristic_scores) * 0.05  # Small noise level
    heuristic_scores += noise

    return heuristic_scores