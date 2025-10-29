import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the heuristic calculations related to current_distance_matrix, delivery_node_demands, and current_load
    # cvrp_scores
    # Generate a modified distance-based heuristic score matrix by emphasizing shorter distances
    modified_distance_scores = -current_distance_matrix * 0.8 + torch.randn_like(current_distance_matrix) * 0.5

    # Generate a modified demand-based heuristic score matrix with a focus on low-demand nodes and randomness
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.6 + torch.min(
        current_load) / 2 + torch.randn_like(current_distance_matrix) * 0.7

    # Calculate the overall score matrix for edge selection
    overall_scores = modified_distance_scores + modified_demand_scores

    return overall_scores