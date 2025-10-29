import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:

    # Base heuristic cost using the current distance matrix
    heuristic_scores = current_distance_matrix * 1.0

    # Applying constraints for load capacity and demands
    feasible_delivery = (delivery_node_demands <= current_load.unsqueeze(-1)).float()
    heuristic_scores += (1 - feasible_delivery) * 1e6  # Penalize infeasible deliveries

    feasible_pickup = (pickup_node_demands <= current_load_open.unsqueeze(-1)).float()
    heuristic_scores += (1 - feasible_pickup) * 1e6  # Penalize infeasible pickups

    # Consider time window constraints
    earliest_times = time_windows[:, 0].unsqueeze(0)
    latest_times = time_windows[:, 1].unsqueeze(0)
    arrival_time_penalty = (arrival_times < earliest_times).float() * 1e6 + \
                           (arrival_times > latest_times).float() * 1e6
    heuristic_scores += arrival_time_penalty

    # Account for remaining route duration
    duration_penalty = (current_length.unsqueeze(-1) < current_distance_matrix).float() * 1e6
    heuristic_scores += duration_penalty

    # Introduce enhanced randomness for diversity and exploration
    randomness = torch.randn_like(current_distance_matrix) * 0.05
    heuristic_scores += randomness

    return heuristic_scores