import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # New computation for distance-based heuristic score matrix with modified randomness
    distance_scores = -current_distance_matrix / (torch.max(current_distance_matrix) + 1e-6) + torch.randn_like(
        current_distance_matrix) * 0.9

    # New computation for demand-based heuristic score matrix with adjusted emphasis and randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.6

    # Combine the different heuristic scores with modified strategies for balanced exploration
    new_scores = distance_scores + demand_scores

    return new_scores