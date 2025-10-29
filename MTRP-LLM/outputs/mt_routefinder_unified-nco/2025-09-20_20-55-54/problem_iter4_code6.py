import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Normalize demands and loads while avoiding division by zero
    epsilon = 1e-8
    delivery_demand_normalized = delivery_node_demands / (torch.abs(delivery_node_demands).max() + epsilon)
    current_load_normalized = current_load / (torch.abs(current_load).max() + epsilon)
    current_load_open_normalized = current_load_open / (torch.abs(current_load_open).max() + epsilon)

    # Calculate feasibility based on time windows
    time_constraints = (arrival_times <= time_windows[:, 1].unsqueeze(0)) & (arrival_times >= time_windows[:, 0].unsqueeze(0))
    
    # Score computation based on distance, demand satisfaction, and feasibility
    distance_score = -current_distance_matrix
    demand_score = delivery_demand_normalized * (1 - current_load_normalized.unsqueeze(1))
    feasibility_score = torch.where(time_constraints, torch.tensor(1.0, device=current_distance_matrix.device), torch.tensor(-1.0, device=current_distance_matrix.device))

    # Combine scores with controlled randomness
    heuristic_scores = distance_score + demand_score + feasibility_score + torch.randn_like(current_distance_matrix) * 0.05

    # Clamp scores to avoid invalid values
    heuristic_scores = torch.clamp(heuristic_scores, -1e3, 1e3)

    return heuristic_scores