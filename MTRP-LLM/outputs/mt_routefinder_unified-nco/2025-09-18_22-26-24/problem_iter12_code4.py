import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize heuristic indicators
    heuristic_indicators = torch.zeros_like(current_distance_matrix)

    # Calculate basic indicators based on distance and demand feasibility
    feasible_deliveries = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_length.unsqueeze(1) >= current_distance_matrix)
    heuristic_indicators += feasible_deliveries.float() * (1 / (current_distance_matrix + 1e-6))  # Avoid division by zero

    # Time window penalties
    current_time = arrival_times + current_distance_matrix
    time_window_penalties = ((current_time < time_windows[:, 0].unsqueeze(0)) | (current_time > time_windows[:, 1].unsqueeze(0))).float()
    heuristic_indicators -= time_window_penalties * 10  # Strong penalty for time window violations

    # Capacity penalties for pickups
    feasible_pickups = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= delivery_node_demands_open.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0))
    heuristic_indicators += feasible_pickups.float() * 0.5  # Bonus for feasible pickups

    # Introduce dynamic randomness to avoid local optima
    noise_factor = torch.abs(torch.randn_like(heuristic_indicators) * 0.1)
    noise_sign = torch.sign(torch.randn_like(heuristic_indicators))
    random_noise = noise_factor * noise_sign
    heuristic_indicators += random_noise

    # Normalize heuristic indicators
    heuristic_indicators = torch.clamp(heuristic_indicators, min=-10, max=10)

    return heuristic_indicators