import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    mutated_distance_scores = -0.8 * (current_distance_matrix ** 1.3 - current_distance_matrix ** 1.1) * 0.25 - torch.randn_like(current_distance_matrix) * 0.15  # Modified exponent and coefficient for distance score computation

    modified_demand_scores = (current_load.unsqueeze(1) - 1.6 * delivery_node_demands.unsqueeze(0).float()) * 0.22 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - 0.8 * torch.min(current_load_open))) * 0.42 + torch.randn_like(current_distance_matrix) * 0.18)  # Adjusted demand score calculation

    time_score = 0.7 * ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.3) +
                        (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.7))  # Unchanged time score calculation

    adjusted_pickup_load = current_load.unsqueeze(1) + 1.2 * pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 1.8) * 0.08 * (1 - (0.7 * delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1)))  # Unchanged pickup score

    pickup_score = torch.clamp(pickup_score, min=-float('inf'), max=float('inf'))  # Unchanged pickup score clamping

    open_delivery_score = (-current_distance_matrix ** 1.6 / (0.9 * delivery_node_demands_open.unsqueeze(0) + torch.mean(current_load_open) * 1.4)).clamp(min=-1.6, max=1.6) * 0.25  # Unchanged open delivery score calculation

    length_score = (current_length.unsqueeze(1) / (1 + current_distance_matrix)).clamp(min=-0.6, max=0.6) * 0.12  # Unchanged length score calculation

    overall_scores = mutated_distance_scores + modified_demand_scores - time_score + pickup_score + open_delivery_score + length_score

    return overall_scores