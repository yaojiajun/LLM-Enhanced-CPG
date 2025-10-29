import torch
import numpy as np
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # New computation for normalized distance-based heuristic score matrix with modified randomness
    norm_dist_scores = torch.exp(-current_distance_matrix) - torch.rand_like(current_distance_matrix) * 0.6

    # New computation for demand-based heuristic score matrix with added noise
    demand_scores = (delivery_node_demands.unsqueeze(0) + current_load.unsqueeze(1)) * 0.6 + torch.min(delivery_node_demands) / 2 + torch.rand_like(current_distance_matrix) * 0.3

    # Introduce additional noise for exploration and diversity
    extra_noise = torch.rand_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies
    cvrp_scores = norm_dist_scores + demand_scores + extra_noise

    # Keep the calculations related to other inputs unchanged

    # Return the overall scores
    return cvrp_scores