import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:

    # Calculate heuristic scores considering various constraints
    available_capacity = current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)
    available_capacity_open = current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)

    # Time window feasibility
    arrival_window_compliance = (arrival_times <= time_windows[:, 1]) & (arrival_times >= time_windows[:, 0])
    
    # Calculate penalties for time windows not being satisfied
    time_penalties = torch.where(arrival_window_compliance, torch.zeros_like(arrival_window_compliance), 
                                  torch.full_like(arrival_window_compliance, -1000))

    # Calculate distance benefits
    distance_benefit = -current_distance_matrix

    # Aggregate scores
    heuristic_scores = distance_benefit + time_penalties.float()

    # Incorporate pickup capacities
    pickup_capacity = current_load.unsqueeze(-1) >= pickup_node_demands.unsqueeze(0)
    heuristic_scores *= pickup_capacity.float()

    # Enhance randomness in scores
    random_noise = torch.randn_like(heuristic_scores) * 0.1  # Add slight randomness
    
    # Final heuristic scores
    heuristic_scores += random_noise
    
    return heuristic_scores