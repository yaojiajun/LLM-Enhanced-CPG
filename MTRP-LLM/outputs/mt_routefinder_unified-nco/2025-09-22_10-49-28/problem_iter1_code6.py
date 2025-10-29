import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify how distance heuristic scores are computed
    distance_heuristic = -torch.sqrt(current_distance_matrix) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Modify how delivery demand scores are computed
    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Keep the pickup score computation the same as in the original function
    pickup_score = torch.zeros_like(current_distance_matrix)

    # Combine the modified heuristic scores with the kept pickup score
    overall_scores = distance_heuristic + delivery_score + pickup_score

    return overall_scores