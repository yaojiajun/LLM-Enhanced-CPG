import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic score matrix with negative scores
    heuristic_scores = -torch.ones_like(current_distance_matrix)

    # Check capacity constraints for delivery nodes 
    delivery_capacity_mask = (delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)).float()
    
    # Check time window feasibility
    time_window_mask = (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)).float() * \
                       (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float()
    
    # Check duration constraints
    duration_mask = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine the masks to filter feasible moves
    feasible_moves_mask = delivery_capacity_mask * time_window_mask * duration_mask
    
    
    # Calculate effective distance scores with some randomness
    effective_distance_scores = current_distance_matrix * feasible_moves_mask

    # Assign positive scores to feasible moves and introduce randomness
    randomness = torch.rand_like(effective_distance_scores) * feasible_moves_mask
    heuristic_scores += -effective_distance_scores + randomness

    # Compute pickup feasibility similarly
    pickup_capacity_mask = (pickup_node_demands.unsqueeze(0) <= current_load_open.unsqueeze(1)).float()
    pickup_duration_mask = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # We can optimize further by enhancing scores for nodes which are pickups
    pickup_feasibility_mask = pickup_capacity_mask * pickup_duration_mask
    heuristic_scores += pickup_feasibility_mask * (1.0 / (effective_distance_scores + 1e-5))

    return heuristic_scores