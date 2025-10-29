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
    
    # Constants
    epsilon = 1e-8
    inf_mask = float('inf')

    # Time Window Constraints
    earliest_constraints = (arrival_times < time_windows[:, 0]).float()
    latest_constraints = (arrival_times > time_windows[:, 1]).float()
    
    # Calculate Total Demand Constraints
    capacity_validity = ((current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) | 
                         (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))).float()
     
    # Calculate Distance Heuristic as potential scores
    distance_scores = 1 / (current_distance_matrix + epsilon)

    # Adjust scores based on delivery demands happenning before service time
    service_scores = distance_scores * (1 - earliest_constraints - latest_constraints) * capacity_validity
    
    # Add randomness to avoid local minima (controlled randomness)
    random_scores = torch.rand(service_scores.shape, device=service_scores.device) * 0.1
    final_scores = service_scores + random_scores

    # NaN/Inf Clamping
    final_scores = torch.clamp(final_scores, min=-inf_mask, max=inf_mask)

    # Ensure all outputs are finite
    final_scores[~torch.isfinite(final_scores)] = 0.0

    return final_scores