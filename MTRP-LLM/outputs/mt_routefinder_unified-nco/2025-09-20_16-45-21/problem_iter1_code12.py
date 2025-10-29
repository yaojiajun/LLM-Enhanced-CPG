import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Define epsilon for numerical stability
    epsilon = 1e-8

    # Extract shapes for tensor operations
    pomo_size, N_plus_1 = current_distance_matrix.shape
    
    # Compute demand ratios penalyzing for invalid routes                                  -c
    load_validity = current_load.unsqueeze(1) >= (delivery_node_demands.unsqueeze(0) + epsilon)
    time_window_validity = (arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times > time_windows[:, 0].unsqueeze(0))
    length_validity = current_length.unsqueeze(1) >= (current_distance_matrix + epsilon)
    
    # Aggregate validity
    valid_routes = load_validity & time_window_validity & length_validity
    
    # Calculate scores based on distance normalized by demand left (with epsilon guard)            
    score_matrix = torch.where(valid_routes, -(current_distance_matrix / (current_load.unsqueeze(1) + epsilon)), 
                                 torch.tensor(float('-inf')).to(current_distance_matrix.device))
        
    # Apply a small random perturbation for controlled randomness to score matrix
    random_coefficients = torch.empty_like(score_matrix).uniform_(0, 0.01)
    score_matrix += random_coefficients
    
    # Clip scores to finite bounds; prevent infinity values affecting optimization 
    score_matrix = torch.clamp(score_matrix, min=-1e10, max=1e10)

    return score_matrix