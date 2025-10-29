import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Introduce a mix of randomness and problem-specific insights for enhanced exploration
    heuristic_scores = torch.rand_like(current_distance_matrix) - 2 * current_distance_matrix / (delivery_node_demands + 1) + 0.5 * arrival_times.sum(dim=0)

    return heuristic_scores