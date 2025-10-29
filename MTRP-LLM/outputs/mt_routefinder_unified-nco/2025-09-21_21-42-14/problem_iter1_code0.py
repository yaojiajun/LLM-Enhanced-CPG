import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # New heuristics for distance, delivery demands, and current load
    distance_heuristic = (current_distance_matrix / torch.mean(current_distance_matrix)) + torch.randn_like(current_distance_matrix) * 0.5

    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    pickup_score = (pickup_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.4 + torch.max(pickup_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.2

    # Combine the modified heuristic scores
    total_scores = distance_heuristic + delivery_score + pickup_score

    vrptw_scores, vrpb_scores, vrpl_scores, ovtp_scores = ..., ..., ..., ...  # retain the original calculation for other scores

    overall_scores = total_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovtp_scores

    return overall_scores