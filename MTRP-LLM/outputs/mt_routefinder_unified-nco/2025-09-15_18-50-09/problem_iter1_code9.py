import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Number of nodes and trajectories
    pomo_size, num_nodes = current_distance_matrix.shape

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)

    # Capacity constraints for deliveries
    delivery_capacity_issues = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float()
    heuristic_scores -= delivery_capacity_issues * 1000  # Penalize infeasible deliveries

    # Capacity constraints for backhauls
    backhaul_capacity_issues = (pickup_node_demands.unsqueeze(0) > current_load_open.unsqueeze(1)).float()
    heuristic_scores -= backhaul_capacity_issues * 1000  # Penalize infeasible pickups

    # Time window feasibility
    for i in range(pomo_size):
        earliest_times = time_windows[:, 0] - arrival_times[i]
        latest_times = time_windows[:, 1] - arrival_times[i]

        # Check if arrival times violate time windows
        time_window_violations = (earliest_times > 0).float() * (latest_times < 0).float()
        heuristic_scores[i] -= time_window_violations * 1000  # Penalize time window violations

    # Route length constraints
    length_issues = (current_distance_matrix + arrival_times).unsqueeze(0) > current_length.unsqueeze(1).unsqueeze(1)
    heuristic_scores -= length_issues * 1000  # Penalize length issues

    # Compute distance-based scores
    normalized_distances = 1 / (current_distance_matrix + 1e-5)  # Avoid division by zero
    heuristic_scores += normalized_distances  # Favor closer nodes

    # Incorporate randomness to avoid local optima
    random_scores = torch.rand((pomo_size, num_nodes), device=current_distance_matrix.device) * 0.1
    heuristic_scores += random_scores  # Add random noise to promote exploration

    return heuristic_scores