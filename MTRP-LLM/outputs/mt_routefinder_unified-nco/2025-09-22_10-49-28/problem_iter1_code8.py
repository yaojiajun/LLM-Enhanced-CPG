import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Modify the calculation related to 'current_distance_matrix', 'delivery_node_demands', and 'current_load' only
    
    # Customized heuristic calculation for distance matrix
    modified_distance_scores = -current_distance_matrix ** 2 / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5
    
    # Customized heuristic calculation for delivery node demands
    modified_delivery_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.9 + torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.4
    
    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    customized_scores = modified_distance_scores + modified_delivery_scores + enhanced_noise

    # Include the remaining calculations unchanged as per the original function
    
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise
    
    # Refer to the original function for the rest of the calculations
    
    return customized_scores + cvrp_scores