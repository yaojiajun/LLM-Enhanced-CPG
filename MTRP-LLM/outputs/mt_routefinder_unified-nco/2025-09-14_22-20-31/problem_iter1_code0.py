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
    
    # Constants for scaling heuristic scores
    DISTANCE_WEIGHT = 1.0
    DEMAND_WEIGHT = -1.0
    TIME_WINDOW_WEIGHT = -0.5
    LENGTH_WEIGHT = -0.7

    # Calculate basic heuristic scores based on distance
    distance_scores = -current_distance_matrix * DISTANCE_WEIGHT

    # Capacity constraints for delivery and pickups
    # Positive score if delivery demands can be met, negative if not
    delivery_feasibility = (current_load[:, None] - delivery_node_demands[None, :]) >= 0
    delivery_scores = torch.where(delivery_feasibility, 
                                   torch.zeros_like(delivery_feasibility).float(), 
                                   torch.full_like(delivery_feasibility, -float('inf')))
    
    pickup_feasibility = (current_load_open[:, None] + pickup_node_demands[None, :]) <= current_load_open[:, None].max()
    pickup_scores = torch.where(pickup_feasibility, 
                                 torch.zeros_like(pickup_feasibility).float(), 
                                 torch.full_like(pickup_feasibility, -float('inf')))
    
    # Time window considerations
    current_time = arrival_times
    time_scores = torch.zeros_like(current_distance_matrix)

    for i in range(time_windows.shape[0]):
        earliest, latest = time_windows[i]
        time_scores[:, i] = torch.where((current_time[:, i] >= earliest) & (current_time[:, i] <= latest), 
                                         torch.zeros_like(current_time[:, i]), 
                                         -float('inf'))

    # Length constraint
    length_scores = (current_length[:, None] - current_distance_matrix) >= 0
    length_scores = torch.where(length_scores, 
                                 torch.zeros_like(length_scores).float(), 
                                 torch.full_like(length_scores, -float('inf')))
    
    # Combine all scores with weights
    heuristic_scores = (distance_scores + 
                        delivery_scores + 
                        pickup_scores + 
                        time_scores + 
                        length_scores)
    
    # Add randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Introducing a small randomness
    heuristic_scores += randomness

    # Normalize scores to ensure all values are in a feasible range
    heuristic_scores = (heuristic_scores - heuristic_scores.min()) / (heuristic_scores.max() - heuristic_scores.min())

    return heuristic_scores