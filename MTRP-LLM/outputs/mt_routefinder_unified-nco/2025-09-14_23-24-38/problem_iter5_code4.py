import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic scores with a low base score to guide exploration
    heuristic_scores = -torch.ones_like(current_distance_matrix)
    
    # Compute capacity feasibility
    delivery_capacity = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    open_capacity = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Capacity constraints scores: positive where deliveries are feasible
    capacity_scores = delivery_capacity + open_capacity
    
    # Compute time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_window_feasibility = (
        (current_time >= time_windows[:, 0].unsqueeze(0)).float() *
        (current_time <= time_windows[:, 1].unsqueeze(0)).float()
    )
    
    # Performance adaptation: combine capacities and time window feasibility
    feasibility_scores = capacity_scores * time_window_feasibility
    
    # Calculate the distance penalty based on travel costs
    distance_penalty = (1 / (current_distance_matrix + 1e-6)) * feasibility_scores  # Avoid division by zero
    
    # Combine scores: higher scores for feasible paths, including randomness for exploration
    random_factor = torch.rand_like(current_distance_matrix) * (1 - feasibility_scores)
    heuristic_scores = feasibility_scores + distance_penalty + random_factor
    
    # Scale the score dynamically based on the severity of constraints
    constraint_severity = 1.0 / (1.0 + (1 - feasibility_scores.sum(dim=1, keepdim=True)))
    heuristic_scores *= constraint_severity
    
    # Apply duration constraints to further evaluate possible routes
    duration_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    heuristic_scores *= duration_feasibility * (1 + 0.5 * random_factor)  # Further introduce randomness

    return heuristic_scores