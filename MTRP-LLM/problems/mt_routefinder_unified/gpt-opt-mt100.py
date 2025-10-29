import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    distance_heuristic = -torch.sqrt(current_distance_matrix + 1e-6) * 0.7

    remaining_load = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    delivery_score = torch.where(remaining_load >= 0, remaining_load * 1.3, remaining_load * 0.5)

    open_remaining_load = current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)
    open_delivery_score = torch.where(open_remaining_load >= 0, open_remaining_load * 1.2, open_remaining_load * 0.9 + 0.7 * delivery_node_demands_open.unsqueeze(0))

    early_arrival_penalty = torch.clamp(time_windows[:, 0].unsqueeze(0) - arrival_times, min=0) ** 2 * 0.6
    late_arrival_penalty = torch.clamp(arrival_times - time_windows[:, 1].unsqueeze(0), min=0) ** 3 * 0.8
    time_score = - (early_arrival_penalty + late_arrival_penalty)

    # Modified calculation related to pickup_node_demands
    updated_pickup_score = torch.where(pickup_node_demands > 0, pickup_node_demands * 0.8, pickup_node_demands * 0.2)  # Updated pickup score calculation
    pickup_score = updated_pickup_score * 0.1  # Adjusted weight for pickups

    total_score = 0.25 * distance_heuristic + 0.4 * delivery_score + 0.35 * open_delivery_score + 0.2 * time_score + pickup_score

    randomness = torch.randn_like(total_score) * 0.25
    final_score = total_score + randomness

    return final_score