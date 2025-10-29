import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Distance-based modification
    normalized_distance_scores = -torch.sqrt(current_distance_matrix) / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Demand-based modification
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4

    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Group update for codes to be changed/optimized
    vrptw_scores, vrpb_scores, vrpl_scores, ovrp_scores = torch.randn_like(current_distance_matrix), torch.randn_like(current_distance_matrix), torch.randn_like(current_distance_matrix), torch.randn_like(current_distance_matrix)
    
    overall_scores=cvrp_scores+vrptw_scores+vrpb_scores+vrpl_scores+ovrp_scores

    return overall_scores