import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    mutated_distance_scores = -1.3 * (torch.log(current_distance_matrix + 1e-6) * 0.6) + torch.randn_like(current_distance_matrix) * 0.3
    adjusted_capacity = current_load.unsqueeze(1).float() + 1.4 * delivery_node_demands.unsqueeze(0).float() + (pickup_node_demands.unsqueeze(0).float() * 2.0)
    total_capacity = current_load.unsqueeze(1).max() + 1.5
    modified_demand_scores = (adjusted_capacity / total_capacity).clamp(min=0) * 0.9 + \
                             (1.7 / (1 + torch.exp(torch.min(current_load_open) - torch.max(current_load_open))) * 1.2 + torch.randn_like(current_distance_matrix) * 0.04)
    adjusted_pickup_load = current_distance_matrix * 0.9 + current_load.unsqueeze(1).float() + 1.2 * pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + torch.sqrt(current_distance_matrix)) ** 1.4) * 0.9 * (1 - (delivery_node_demands.unsqueeze(0) / (current_load.unsqueeze(1) + 1.5)) ** 1.6)

    open_delivery_score = (1 / (1 + current_load_open.unsqueeze(1) * 0.6 / (1 + delivery_node_demands_open.unsqueeze(0) + 1e-5))) * 1.3 * (1 / (1 + torch.exp(torch.min(delivery_node_demands_open) - torch.max(delivery_node_demands_open))) ** 1.7 + torch.randn_like(current_distance_matrix) * 0.1)

    modified_arrival_delay = arrival_times - (time_windows[:, 0].unsqueeze(0) + 0.09 * torch.sqrt(torch.abs(torch.randn_like(current_distance_matrix))))

    time_score = 0.85 * (torch.max(modified_arrival_delay, torch.zeros_like(modified_arrival_delay)) * 0.75 +
                     torch.abs(torch.min(modified_arrival_delay, torch.zeros_like(modified_arrival_delay)) * 0.25))

    length_penalty = ((current_length.unsqueeze(1) - current_distance_matrix) / (1 + current_distance_matrix + 1e-6)).clamp(max=1) * 1.3
    length_score = 0.45 * (1 - length_penalty) * 1.8  # Modified the calculation for length_score

    overall_scores = mutated_distance_scores + modified_demand_scores - time_score + pickup_score + open_delivery_score - length_score

    return overall_scores