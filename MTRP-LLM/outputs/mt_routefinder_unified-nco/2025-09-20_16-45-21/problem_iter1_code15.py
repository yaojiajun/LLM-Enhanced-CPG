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
    epsilon = 1e-8  # Small value to prevent div by zero in calculations
    batch_size, num_nodes = current_distance_matrix.shape
    
    # Calculate effects of distance - penalize longer distances with inverse scaling
    distance_scores = 1.0 / (current_distance_matrix + epsilon)

    # Delivery capacity availability - channel vehicle's remaining load
    capacity_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) / (current_load.unsqueeze(1) + epsilon)

    # Opening routes availability check
    open_capacity_scores = (current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)) / (current_load_open.unsqueeze(1) + epsilon)

    # Time windows evaluation - calculate feasibility
    earlies = time_windows[:, 0].unsqueeze(0)  # (1, N+1)
    lateness = time_windows[:, 1].unsqueeze(0)  # (1, N+1)
    
    time_feasibility_scores = torch.where((arrival_times < lateness) & (arrival_times >= earlies), 
                                          torch.tensor(1.0, device=current_distance_matrix.device), 
                                          torch.tensor(0.0, device=current_distance_matrix.device))
    
    # Route duration check - remain under the preset thresholds
    duration_scores = (current_length.unsqueeze(1) - current_distance_matrix) / (current_length.unsqueeze(1) + epsilon)

    # Control randomness in scores - perturb, add stochasticity component
    random_component = torch.rand_like(current_distance_matrix) * 0.01  # Introduce randomness
    
    # Compute final heuristics score
    scores = (distance_scores * capacity_scores * open_capacity_scores * 
              time_feasibility_scores * duration_scores) + random_component
    
    # Ensure only finite values are accounted
    scores = torch.clamp(scores, min=-torch.inf, max=torch.inf)

    return scores