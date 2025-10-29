import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Calculate capacity feasibility
    delivery_capacity_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands).float()
    pickup_capacity_feasibility = (current_load_open.unsqueeze(1) >= pickup_node_demands).float()
    
    # Evaluate time window feasibility
    time_window_feasibility = ((arrival_times < time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times > time_windows[:, 0].unsqueeze(0))).float()
    
    # Combine constraints into a single feasibility matrix
    feasibility_matrix = delivery_capacity_feasibility * pickup_capacity_feasibility * time_window_feasibility
    
    # Compute earning potentials based on deliveries and pickups
    earning_potentials = (delivery_node_demands.unsqueeze(0) + pickup_node_demands.unsqueeze(0)) * feasibility_matrix
    
    # Weighted distance scores with potential earnings and constraints
    distance_weights = current_distance_matrix * (1 - feasibility_matrix)  # Penalize infeasible routes
    weighted_scores = distance_weights - earning_potentials
    
    # Add enhanced randomness to the scores for diversity
    randomness = 0.1 * torch.randn_like(weighted_scores)
    heuristic_scores = weighted_scores + randomness
    
    return heuristic_scores