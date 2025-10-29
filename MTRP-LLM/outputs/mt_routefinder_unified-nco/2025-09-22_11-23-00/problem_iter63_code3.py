import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    mutated_distance_scores = -1 * (current_distance_matrix ** 1.3) * 0.3 - torch.randn_like(current_distance_matrix) * 0.2

    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.2 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.4 + torch.randn_like(current_distance_matrix) * 0.2)

    time_score = 0.7 * ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.3) +
                        (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.7))

    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 2) * 0.15 * (1 - (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1)))  # Modified logic for pickup score

    pickup_score = torch.clamp(pickup_score, min=-float('inf'), max=float('inf'))

    open_delivery_score = (-current_distance_matrix ** 1.2 / (delivery_node_demands_open.unsqueeze(0) + torch.mean(current_load_open)*1.2)).clamp(min=-2, max=2) * 0.25  # Adjusted load balancing factor in open_delivery_score

    length_score = (current_length.unsqueeze(1) + current_distance_matrix).clamp(min=0, max=1) * 0.25  # Modified calculation for length_score

    overall_scores = mutated_distance_scores + modified_demand_scores - time_score + pickup_score + open_delivery_score + length_score

    return overall_scores