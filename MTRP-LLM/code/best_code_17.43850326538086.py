import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    mutated_distance_scores = -1.2 * torch.sin(current_distance_matrix * 0.8) + torch.randn_like(current_distance_matrix) * 0.3
    adjusted_delivery_demand = delivery_node_demands.unsqueeze(0).float() * 2.0 + 0.6 * current_load.unsqueeze(1).float()
    delivery_score = (current_load.unsqueeze(1) / (adjusted_delivery_demand + 2.0)) * 0.5

    adjusted_load = current_load.unsqueeze(1).float() + 0.6 * delivery_node_demands.unsqueeze(0).float() - 0.8 * current_distance_matrix
    load_adjustment_penalty = (adjusted_load / adjusted_load.max(dim=1).values.unsqueeze(1)).clamp(max=1) * 0.25

    open_delivery_score = (current_load_open.unsqueeze(1) / (delivery_node_demands_open.unsqueeze(0) + 2.2)) * 0.2

    # Modify the calculation related to 'current_length' for enhanced scoring
    length_score = current_length.unsqueeze(1) / (torch.sqrt(current_distance_matrix) + 0.7) * 0.12

    return mutated_distance_scores + delivery_score - load_adjustment_penalty + open_delivery_score + length_score