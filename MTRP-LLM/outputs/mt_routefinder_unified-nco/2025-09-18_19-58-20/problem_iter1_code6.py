import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic score matrix
    pomo_size, num_nodes = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)
    
    # Calculate feasible delivery nodes based on current load and demands
    delivery_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Calculate waiting times at nodes based on time windows
    wait_times = torch.maximum(torch.zeros_like(arrival_times), time_windows[:, 0] - arrival_times)
    is_within_time_window = (arrival_times <= time_windows[:, 1])
    
    # Calculate time window penalties
    time_window_penalty = -1000 * (wait_times + ~is_within_time_window.float())
    
    # Distance penalties are merged with delivery feasibility
    distance_penalty = -current_distance_matrix * delivery_feasible.float()
    
    # Compute total heuristic scores considering distance, feasibility, and time windows
    heuristic_scores += distance_penalty + time_window_penalty

    # Capacity constraints handling for pickups
    pickup_feasible = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= current_load_open.unsqueeze(1))
    pickup_penalty = -10 * ~pickup_feasible.float()
    
    heuristic_scores += pickup_penalty
    
    # Introduce randomness to enhance exploration of edges
    randomness = torch.rand((pomo_size, num_nodes), device=current_distance_matrix.device) * 0.1 - 0.05
    heuristic_scores += randomness
    
    return heuristic_scores