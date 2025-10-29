import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Update the distance-based heuristic score matrix with emphasis on long distances
    distance_scores = -current_distance_matrix * 0.6 + torch.randn_like(current_distance_matrix) * 0.3

    # Adjust the demand-based heuristic score matrix with a focus on high-demand nodes and randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.4 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5
   
    # Calculate the overall score matrix for edge selection
    overall_scores = distance_scores + demand_scores

    return overall_scores