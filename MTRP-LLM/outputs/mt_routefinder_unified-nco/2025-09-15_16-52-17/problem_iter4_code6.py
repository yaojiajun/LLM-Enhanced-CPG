import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Implement further enhanced heuristics logic with adaptive penalties, exploration, and exploitation
    # Calculate heuristic score matrix with adaptive penalties, exploration, and exploitation mechanisms
    exploration_factor = torch.rand_like(current_distance_matrix)  # Introduce exploration factor
    exploitation_factor = current_distance_matrix.mean(1, keepdim=True) / current_distance_matrix.std(1, keepdim=True)  # Introduce exploitation factor
    heuristic_scores = exploration_factor * 0.5 + exploitation_factor * 0.5  # Combined exploration and exploitation scores

    return heuristic_scores