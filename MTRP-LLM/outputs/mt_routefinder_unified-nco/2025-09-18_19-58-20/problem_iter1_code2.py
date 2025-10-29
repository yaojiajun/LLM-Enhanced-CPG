import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor,
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor,
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Capacity Constraints for Delivery
    can_deliver = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Capacity Constraints for Open Routes
    can_deliver_open = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Time Window Violation
    earliest_arrivals = torch.max(arrival_times, time_windows[:, 0].unsqueeze(0))
    within_time_window = (earliest_arrivals <= time_windows[:, 1].unsqueeze(0)).float()
    
    # Route Length Constraints
    length_constraints = (current_length.unsqueeze(-1) >= current_distance_matrix).float()

    # Calculate base scores from the distance matrix
    distance_scores = -current_distance_matrix
    
    # Combine scores with constraints
    heuristic_scores = distance_scores + (can_deliver * can_deliver_open * within_time_window * length_constraints)
    
    # Introduce randomness for avoiding local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.1
    heuristic_scores += random_noise
    
    return heuristic_scores