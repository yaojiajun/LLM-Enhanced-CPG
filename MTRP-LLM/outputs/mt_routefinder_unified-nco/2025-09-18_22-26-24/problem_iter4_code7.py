import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Enhanced randomness by introducing noise with variable scale factor based on delivery demands
    noise_scale = 0.1 * delivery_node_demands.unsqueeze(0)  # Adding noise with a scale factor based on delivery demands
    noise = torch.rand_like(current_distance_matrix) * noise_scale
    heuristic_indicators = torch.rand_like(current_distance_matrix) + noise

    # Additional innovative heuristics can be implemented here

    return heuristic_indicators