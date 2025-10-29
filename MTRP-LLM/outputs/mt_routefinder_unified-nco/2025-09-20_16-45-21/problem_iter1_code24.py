import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Constants
    epsilon = 1e-8

    # Calculate availability for deliveries based on current load for both types (standard and open routes)
    delivery_feasibility = (current_load.unsqueeze(1) - delivery_node_demands).clamp(min=0)
    open_delivery_feasibility = (current_load_open.unsqueeze(1) - delivery_node_demands_open).clamp(min=0)
    
    # Check time window feasibilities
    earliest_times = time_windows[:, 0].unsqueeze(0)
    latest_times = time_windows[:, 1].unsqueeze(0)
    arrivalFeasibility = (arrival_times < latest_times + epsilon) & (arrival_times > earliest_times - epsilon)
    
    # Compute score matrix
    scores = torch.zeros_like(current_distance_matrix)

    # Adding logistic score for distances (small distances should yield large scores)
    dist_score = 1 / (current_distance_matrix + epsilon)
    
    # Combine all constraints into scores
    scores += dist_score
    scores += (delivery_feasibility > 0).float()  # Add credits for feasible deliveries
    scores += (open_delivery_feasibility > 0).float()  # Add credits for feasible open deliveries
    scores += arrivalFeasibility.float()  # Add credits for feasible arrival times
    
    # Introduce random noise for score diffusion
    random_noise = torch.randn_like(scores) * 0.01  # Small random noise
    scores += random_noise

    # Clamp scores to ensure numerical stability
    scores = torch.clamp(scores, min=float('-inf'), max=float('inf'))

    return scores