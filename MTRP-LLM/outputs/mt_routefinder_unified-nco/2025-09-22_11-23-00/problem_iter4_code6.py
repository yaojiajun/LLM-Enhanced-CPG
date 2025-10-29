import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    distance_heuristic = -1 * (current_distance_matrix ** 1.1) * 0.28 + torch.randn_like(current_distance_matrix) * 0.22
    delivery_score = (current_load.unsqueeze(1) - 1.1*delivery_node_demands.unsqueeze(0).float()) * 0.18 + \
                     (1 / (1 + 1.5 * torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.25 + torch.randn_like(current_distance_matrix) * 0.2)

    pickup_score = (current_load.unsqueeze(1) / ((1.05 + current_distance_matrix) ** 1.1)) * 0.20

    mutated_demand_scores = (current_distance_matrix ** 1.12) * 0.22 + torch.randn_like(current_distance_matrix) * 0.18
    modified_demand_scores = (mutated_demand_scores * current_load.unsqueeze(1) / (1.1 + delivery_node_demands.unsqueeze(0).float())).clamp(min=-0.7, max=0.7) * 0.25

    time_score = 0.7 * ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.3) +
                        (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.7))

    open_delivery_score = (-1.3 * current_distance_matrix + 1.3) / (delivery_node_demands_open.unsqueeze(0) + torch.mean(current_load_open)).clamp(0.2, 4) * 0.27

    length_score = (current_length.unsqueeze(1) / (1.1 + current_distance_matrix)).clamp(min=-0.7, max=0.7) * 0.23

    overall_scores = distance_heuristic + delivery_score - pickup_score + modified_demand_scores - time_score + open_delivery_score + length_score

    return overall_scores