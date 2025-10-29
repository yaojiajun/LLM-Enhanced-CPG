import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    mutated_distance_scores = -1 * (current_distance_matrix ** 1.3) * 0.3 - torch.randn_like(current_distance_matrix) * 0.2

    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.2 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.4 + torch.randn_like(current_distance_matrix) * 0.2)

    # Updated time_score calculation
    time_penalty = (arrival_times - time_windows[:, 1].unsqueeze(0)).clamp(min=0) * 0.5   # Penalty if arrival time exceeds the latest window
    time_bonus = (time_windows[:, 0].unsqueeze(0) - arrival_times).clamp(min=0) * 0.3      # Bonus if arrival is before the earliest window
    time_score = -time_penalty + time_bonus

    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 2) * 0.15 * (1 - (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1)))  # Modified logic for pickup score

    delivery_score_mod = (-current_distance_matrix ** 1.2 / (delivery_node_demands_open.unsqueeze(0) * 2 + torch.mean(current_load_open) * 1.5)).clamp(min=-2, max=2) * 0.25  # Updated delivery score calculation based on delivery_node_demands_open and current_load_open

    length_score = (current_length.unsqueeze(1) / (1 + current_distance_matrix)).clamp(min=-1, max=1) * 0.15

    overall_scores = mutated_distance_scores + modified_demand_scores + time_score + pickup_score + delivery_score_mod + length_score

    return overall_scores