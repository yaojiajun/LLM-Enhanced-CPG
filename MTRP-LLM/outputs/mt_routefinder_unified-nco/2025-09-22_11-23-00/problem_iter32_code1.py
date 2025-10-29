import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    mutated_distance_scores = -1 * (current_distance_matrix ** 1.3) * 0.3 - torch.randn_like(current_distance_matrix) * 0.2

    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.2 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.4 + torch.randn_like(current_distance_matrix) * 0.2)

    early_arrival_bonus = (time_windows[:, 0].unsqueeze(0) - arrival_times) * (arrival_times < time_windows[:, 0].unsqueeze(0)).float() * 0.6  # Stronger reward for early arrivals
    late_arrival_penalty = (arrival_times - time_windows[:, 1].unsqueeze(0)) * (arrival_times > time_windows[:, 1].unsqueeze(0)).float() * 1.0  # Stronger penalty for late arrivals
    time_score = early_arrival_bonus - late_arrival_penalty  # Combine bonuses and penalties

    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 2) * 0.15 * (1 - (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1))) 

    open_delivery_score = (-current_distance_matrix ** 1.2 / (delivery_node_demands_open.unsqueeze(0) + torch.mean(current_load_open)*1.2)).clamp(min=-2, max=2) * 0.25  

    length_score = (current_length.unsqueeze(1) / (1 + current_distance_matrix)).clamp(min=-1, max=1) * 0.15

    overall_scores = mutated_distance_scores + modified_demand_scores + time_score + pickup_score + open_delivery_score + length_score

    return overall_scores