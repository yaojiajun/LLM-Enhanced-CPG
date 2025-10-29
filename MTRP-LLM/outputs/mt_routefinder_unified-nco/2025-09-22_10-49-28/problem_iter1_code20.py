import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute a modified normalized distance-based heuristic score matrix with altered calculation approach
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5  # Modified randomness factor

    # Adjust demand-based heuristic score matrix calculation with different weights and noise levels
    # Emphasizing more on current load
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        current_load) * 0.4 + torch.randn_like(current_distance_matrix) * 0.4  # Modified weights and noise level

    # Introduce modified noise factor for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5  # Altered noise level

    # Combine the different heuristic scores with altered strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keep the calculation for the remaining parts identical from the original function

    return cvrp_scores