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

    # Initialize heuristic scores with a base score influenced by the distance matrix
    heuristic_scores = -current_distance_matrix.clone()
    
    # Incorporate delivery node demands and current load constraints
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0))
    heuristic_scores += feasible_delivery.float() * 10  # Weight for feasible delivery
    
    # Incorporate time window feasibility
    current_time = arrival_times.clone()
    for i in range(current_time.size(0)):
        current_time[i] = torch.clamp(current_time[i], time_windows[:, 0], time_windows[:, 1])
    
    time_window_feasible = (current_time >= time_windows[:, 0]) & (current_time <= time_windows[:, 1])
    heuristic_scores += time_window_feasible.float() * 5  # Weight for time window feasibility
    
    # Incorporate backhaul capacity evaluations
    feasible_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0))
    heuristic_scores += feasible_pickup.float() * 7  # Weight for feasible pickups
    
    # Incorporate current length constraints
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)
    heuristic_scores += length_feasibility.float() * 8  # Weight for length feasibility

    # Add a randomness component to diversify exploration
    random_factor = torch.rand_like(heuristic_scores) * 0.5  # Randomness to avoid local optima
    heuristic_scores += random_factor

    return heuristic_scores