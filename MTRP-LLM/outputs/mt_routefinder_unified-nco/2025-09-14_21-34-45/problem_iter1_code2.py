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
    
    # Get the number of vehicles and nodes
    pomo_size, N_plus_1 = current_distance_matrix.shape
    
    # Initialize heuristic score matrix
    heuristics_scores = torch.zeros(pomo_size, N_plus_1, device=current_distance_matrix.device)

    # Calculate capacities and time windows compliance
    delivery_compliance = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()  # Vehicle capacity for deliveries
    open_delivery_compliance = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()  # Open vehicle capacity
    time_window_compliance = ((arrival_times.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0)) & 
                               (arrival_times.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0))).float()  # Time windows

    # Calculate length compliance
    length_compliance = (current_length.unsqueeze(1) >= current_distance_matrix).float()  # Route length budget
    
    # Combine constraints into a compliance score
    compliance_score = delivery_compliance * time_window_compliance * length_compliance * open_delivery_compliance

    # Create a score based on distance and compliance
    distance_score = -current_distance_matrix  # Negative distance - we want to minimize

    # Combine distance and compliance scores
    heuristics_scores = distance_score + compliance_score * 1000  # Weighted score

    # Introduce randomness to escape local optima
    randomness = torch.rand_like(heuristics_scores) * 0.1  # small random noise
    heuristics_scores += randomness

    return heuristics_scores