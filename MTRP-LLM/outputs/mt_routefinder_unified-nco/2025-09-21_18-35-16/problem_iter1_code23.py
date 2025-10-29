import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    distance_heuristic = -current_distance_matrix / torch.max(current_distance_matrix, dim=1, keepdim=True).values + torch.randn_like(current_distance_matrix) * 0.7

    delivery_score = ((delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) / (torch.max(delivery_node_demands) + 1e-8)) * 0.8 + torch.randn_like(current_distance_matrix) * 0.5

    demand_noise = torch.randn_like(current_distance_matrix) * 2.0

    new_scores = distance_heuristic + delivery_score + demand_noise

    return new_scores