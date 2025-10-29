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
    
    # Calculate unused capacities for deliveries and pickups
    delivery_capacities = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)  # (pomo_size, N+1)
    pickup_capacities = current_load_open.unsqueeze(1) - pickup_node_demands.unsqueeze(0)  # (pomo_size, N+1)

    # Check feasibility of visiting nodes regarding capacities
    delivery_feasible = (delivery_capacities >= 0).float()
    pickup_feasible = (pickup_capacities >= 0).float()
    
    # Evaluate time window constraints
    current_time = arrival_times  # (pomo_size, N+1)
    time_window_start = time_windows[:, 0].unsqueeze(0)  # (1, N+1)
    time_window_end = time_windows[:, 1].unsqueeze(0)  # (1, N+1)
    
    time_window_feasibility = ((current_time >= time_window_start) & (current_time <= time_window_end)).float()
    
    # Calculate remaining length available for routes
    remaining_length_matrix = current_length.unsqueeze(1) - current_distance_matrix  # (pomo_size, N+1)
    length_feasibility = (remaining_length_matrix >= 0).float()
    
    # Aggregate feasibility scores
    feasibility_matrix = delivery_feasible * pickup_feasible * time_window_feasibility * length_feasibility  # (pomo_size, N+1)
    
    # Calculate heuristic scores based on distance and feasibility
    heuristic_scores = feasibility_matrix * (1.0 / (current_distance_matrix + 1e-5))  # Avoid division by zero
    
    # Introduce randomness in the heuristic scores to escape local optima
    random_noise = torch.rand_like(heuristic_scores) * 0.1  # Small random perturbation
    heuristic_scores += random_noise

    return heuristic_scores