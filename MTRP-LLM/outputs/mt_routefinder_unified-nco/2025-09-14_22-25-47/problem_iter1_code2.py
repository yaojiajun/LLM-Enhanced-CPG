import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate load feasibility for delivery nodes
    load_feasibility_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0))
    load_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Calculate time window feasibility
    time_window_feasibility = (arrival_times.unsqueeze(1) < time_windows[:, 1].unsqueeze(0)) & \
                               (arrival_times.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0))
    
    # Adjust scores based on distance and feasibility of delivery
    feasible_score = load_feasibility_delivery & time_window_feasibility
    scores[feasible_score] = 1 - (current_distance_matrix[feasible_score] / current_distance_matrix.max())  # Inverse scaling of distances
    
    # Adjust scores based on length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)
    scores[length_feasibility] += 0.5  # Add additional weight for length feasibility

    # Incorporate randomness to avoid local optima
    randomness = torch.rand_like(scores) * 0.1  # Small random values to the scores
    scores += randomness

    # Penalize based on infeasibility for nodes that are not reachable
    scores[~(feasible_score | length_feasibility)] = -1  # Use -1 as a penalty for undesirable edges
    
    return scores