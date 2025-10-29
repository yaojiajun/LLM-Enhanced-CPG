import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score with random noise addition
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score with added randomness
    demand_scores = (delivery_node_demands ** 1.2 - current_load) * 0.5 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise
    
    # Keep the rest of the code unchanged from the original heuristics function
    
    return cvrp_scores