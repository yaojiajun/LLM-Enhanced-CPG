import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the distance heuristic calculation using exponential function for diversity
    exponential_distance_scores = torch.exp(-current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand score calculation by emphasizing high-demand nodes with a square function
    demand_scores = (delivery_node_demands.unsqueeze(0)**2 - current_load.unsqueeze(1)**2) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce the random component for increased randomness
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the modified heuristic scores for balanced exploration
    modified_cvrp_scores = exponential_distance_scores + demand_scores + enhanced_noise

    # Keep the logic for the remaining parts intact
    
    return modified_cvrp_scores