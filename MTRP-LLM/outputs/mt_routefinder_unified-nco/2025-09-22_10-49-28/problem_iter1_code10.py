import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the distance-based heuristic calculation with custom adjustments
    distance_heuristic = (-current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(current_distance_matrix) * 0.7) * 2.0

    # Modify the demand-based heuristic calculation with increased weight and noise level
    demand_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 1.0 + torch.max(delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.8

    # Combine the different heuristic scores with customized adjustments
    overall_scores = distance_heuristic + demand_score

    return overall_scores