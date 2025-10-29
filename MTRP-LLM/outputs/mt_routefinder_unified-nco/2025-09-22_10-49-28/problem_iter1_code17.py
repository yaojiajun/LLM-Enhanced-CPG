import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores modification
    normalized_distance_scores = torch.randn_like(current_distance_matrix) * 0.7  # Adding randomness

    # Revising demand-based score computation for exploration 
    demand_scores = torch.randn_like(current_distance_matrix) * 0.5
    
    # Introducing different noise for diversification
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # Keeping other parts as in the original heuristics function

    return cvrp_scores