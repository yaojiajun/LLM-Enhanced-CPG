import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Introduce enhanced randomness and adaptively weighted distances
    random_scores = torch.randn_like(current_distance_matrix) * 0.5
    distance_based_scores = current_distance_matrix * (torch.rand_like(current_distance_matrix) * 0.5 + 0.5)

    # Balance exploration and exploitation by combining the scores
    heuristic_scores = distance_based_scores + random_scores

    return heuristic_scores