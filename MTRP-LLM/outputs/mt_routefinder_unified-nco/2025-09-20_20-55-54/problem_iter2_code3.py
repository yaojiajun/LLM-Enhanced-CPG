import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Apply noise to distances
    noise = torch.rand_like(current_distance_matrix) * 1e-6
    perturbed_distances = current_distance_matrix + noise

    # Calculate heuristic score based on normalized and perturbed distances with a combination of randomness and distance inverse
    heuristic_scores = (2 * torch.rand_like(current_distance_matrix) - 1) / (perturbed_distances + 1e-8)  # Random scores between -1 and 1

    return heuristic_scores