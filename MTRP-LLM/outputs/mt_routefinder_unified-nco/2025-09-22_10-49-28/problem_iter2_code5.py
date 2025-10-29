import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
  
    # Modified calculations for 'current_distance_matrix', 'delivery_node_demands', and 'current_load'
    # Revised distance-based heuristic score matrix with absolute differences and increased noise
    distance_diff_scores = -torch.abs(current_distance_matrix - torch.mean(current_distance_matrix, axis=1, keepdim=True)) + torch.randn_like(current_distance_matrix) * 0.6

    # Adjusted demand-based heuristic score matrix with load-demand ratios and noise
    demand_ratios = current_load.unsqueeze(1) / (delivery_node_demands + 1e-8)
    demand_scores = demand_ratios * 0.5 - torch.randn_like(current_distance_matrix) * 0.4

    # Combine the modified distance and demand scores with noise for exploration
    combined_scores = distance_diff_scores + demand_scores + 0.2 * torch.randn_like(current_distance_matrix)

    return combined_scores