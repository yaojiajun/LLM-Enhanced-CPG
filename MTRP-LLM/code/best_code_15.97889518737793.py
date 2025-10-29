import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    mutated_distance_scores = -1.6 * torch.sin(current_distance_matrix * 1.4) + torch.randn_like(current_distance_matrix) * 0.3
    adjusted_delivery_demand = delivery_node_demands.unsqueeze(0).float() + 1.4 * current_load.unsqueeze(1).float()
    delivery_score = (current_load.unsqueeze(1) / (adjusted_delivery_demand + 1.2)) * 0.5
    adjusted_load = current_load.unsqueeze(1).float() - 0.7 * current_distance_matrix + 0.9 * delivery_node_demands.unsqueeze(0).float()
    load_adjustment_penalty = (adjusted_load / adjusted_load.max(dim=1).values.unsqueeze(1)).clamp(max=0.8) * 0.6
    
    open_delivery_score_v2 = torch.sqrt(torch.abs(current_load_open.unsqueeze(1) - 1.6 * delivery_node_demands_open.unsqueeze(0))) - 0.6 * torch.sin(current_distance_matrix)
    
    length_score_v3 = current_length.unsqueeze(1) * torch.cos(current_distance_matrix) - torch.sqrt(current_distance_matrix) * 0.7
    
    pickup_demand_factor_v3 = 0.3 * pickup_node_demands.unsqueeze(0).float() + 0.9 * current_load.unsqueeze(1).float()
    pickup_score_v3 = (current_load.unsqueeze(1) / (pickup_demand_factor_v3 + 1.4)) * 0.6

    arrival_difference_start = arrival_times - time_windows[:, 0].unsqueeze(0)
    arrival_penalty_start = arrival_difference_start * 0.7
    time_score_v6 = 1.4 - arrival_penalty_start

    modified_score = mutated_distance_scores + delivery_score - load_adjustment_penalty + open_delivery_score_v2 + time_score_v6 + length_score_v3 - pickup_score_v3

    return modified_score