import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    mutated_distance_scores = -1 * (current_distance_matrix ** 1.4) * 0.4 - torch.randn_like(current_distance_matrix) * 0.3

    modified_demand_scores = (current_load.unsqueeze(1) + delivery_node_demands.unsqueeze(0).float()) * 0.3 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.5 + torch.randn_like(current_distance_matrix) * 0.1)

    time_score = 0.6 * ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.4) +
                        (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.6))

    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 1.3) * 0.15 * (1 - (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1)) ** 1.1)

    open_delivery_score = -torch.abs(current_load_open.unsqueeze(1) - delivery_node_demands_open) * 0.1
    delivery_score_mod = (-current_distance_matrix ** 1.4 / (delivery_node_demands_open.unsqueeze(0) * 1.6 + current_load_open.unsqueeze(1) * 0.9)).clamp(min=-2, max=2) * 0.3
    delivery_score_mod += open_delivery_score

    length_score = (current_length.unsqueeze(1) - current_distance_matrix) * 0.25 / (1 + current_distance_matrix)
    length_score = length_score.clamp(min=-0.6, max=0.6) * 0.35

    overall_scores = mutated_distance_scores + modified_demand_scores - time_score + pickup_score + delivery_score_mod + length_score

    return overall_scores