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
    
    # Initialize scores matrix
    scores = torch.zeros_like(current_distance_matrix)

    # Calculate potential feasible visits based on load capacities
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float()

    # Check time window feasibility
    arrival_time_feasible = (arrival_times < time_windows[:, 1].unsqueeze(0)).float()
    waiting_time = torch.clamp(time_windows[:, 0].unsqueeze(0) - arrival_times, min=0)  # Waiting time if early
    time_window_score = arrival_time_feasible * (1 - (waiting_time / 10.0))  # Normalize waiting time impact

    # Incorporate route duration constraints
    route_length_feasible = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Compute heuristic scores
    scores += feasible_delivery * time_window_score * route_length_feasible
    scores -= current_distance_matrix / 100.0  # Penalize larger distances

    # Introduce randomness to enhance exploration
    randomness = torch.rand_like(scores) * 0.1  # Small random noise
    scores += randomness

    # Apply penalties for infeasible paths
    scores *= (feasible_delivery * feasible_pickup * route_length_feasible)

    return scores