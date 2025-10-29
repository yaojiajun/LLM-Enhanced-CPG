import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # cvrp_scores_v2
    # Modify the normalized distance-based heuristic score with different scaling and noise level
    normalized_distance_scores_v2 = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score with a new formula and increased randomness
    demand_scores_v2 = (current_load.unsqueeze(0) - delivery_node_demands.unsqueeze(1)) * 1.2 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.7

    # Adding a new functionality to incorporate a bias term based on current_load
    bias_term = current_load / (delivery_node_demands + 1e-8)
    
    # Combine modified heuristic scores with added bias for a new cvrp_scores version
    cvrp_scores_v2 = normalized_distance_scores_v2 * 0.6 + demand_scores_v2 * 0.4 + bias_term

    # Rest of the code remains unchanged

    return cvrp_scores_v2