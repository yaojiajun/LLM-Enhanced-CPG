import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score with a different approach
    distance_heuristic = current_distance_matrix / (torch.min(current_distance_matrix) + 1e-8) + torch.randn_like(current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score by adding a dynamic component
    delivery_score = (current_load - delivery_node_demands.unsqueeze(0)) * 0.5 + torch.max(delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    cvrp_scores = distance_heuristic + delivery_score

    # Keep vrptw_scores, vrpb_scores, vrpl_scores, and ovrp_scores computations unchanged

    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores