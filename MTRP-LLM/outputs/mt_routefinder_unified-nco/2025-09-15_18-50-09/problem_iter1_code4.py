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
    
    # Initialize the heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate time feasibility
    earliest_arrival = arrival_times + current_distance_matrix
    time_waiting = torch.clamp(time_windows[:, 0] - earliest_arrival, min=0)
    
    # Penalize for waiting time
    heuristic_scores += -time_waiting * 10  # Weight waiting time penalty
    
    # Capacity constraints for deliveries
    can_deliver = (current_load.unsqueeze(1) >= delivery_node_demands)
    heuristic_scores += can_deliver.float() * 10  # Positive score for feasible deliveries
    
    # Capacity constraints for pickups (when open)
    can_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands)
    heuristic_scores += can_pickup.float() * 5  # Positive score for feasible pickups

    # Reduce score for exceeding capacity
    exceeded_capacity = (current_load.unsqueeze(1) + delivery_node_demands > current_load.max())
    heuristic_scores += -exceeded_capacity.float() * 10  # Heavy penalty for exceeding capacity
    
    # Duration constraints
    within_duration = (current_length.unsqueeze(1) >= current_distance_matrix)
    heuristic_scores += within_duration.float() * 5  # Positive score for feasible routes
    
    # Add randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 2 - 1  # Random values between -1 and 1
    heuristic_scores += randomness * 0.5  # Moderation of random influence
    
    return heuristic_scores