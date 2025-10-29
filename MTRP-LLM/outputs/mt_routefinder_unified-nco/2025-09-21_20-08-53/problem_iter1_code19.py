import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Change the computation of normalized distance-based heuristic score with different scaling and noise level
    normalized_distance_scores = -(current_distance_matrix + torch.min(current_distance_matrix)) / torch.max(current_distance_matrix) * 0.8 + torch.randn_like(current_distance_matrix) * 0.5

    # Change the computation of demand-based heuristic score with modified emphasis on high-demand nodes and randomness level
    demand_scores = (delivery_node_demands * 0.2 - current_load) * 0.7 + torch.max(delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    # Introduce a new form of noise addition for exploration
    new_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with altered strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + new_noise

    # Keep the calculations for other parts unchanged

    return cvrp_scores