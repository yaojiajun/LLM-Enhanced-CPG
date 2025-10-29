import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    pomo_size, N_plus_1 = current_distance_matrix.shape
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate random noise for exploration
    noise = torch.rand_like(current_distance_matrix) * 0.1  # Small random noise for dynamic exploration
    
    # Assess capacity constraints
    feasible_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Assess time window feasibility
    arrival_at_node = arrival_times + current_distance_matrix  # Estimated arrival at nodes
    feasible_time_windows = (arrival_at_node >= time_windows[:, 0].unsqueeze(0)) & (arrival_at_node <= time_windows[:, 1].unsqueeze(0))
    
    # Evaluate route length constraints
    feasible_length = (current_length.unsqueeze(1) >= current_distance_matrix.sum(dim=1).unsqueeze(0))
    
    # Combine constraints to find feasible nodes
    feasible_nodes = feasible_delivery & feasible_time_windows & feasible_length
    
    # Calculate heuristic indicators based on feasible routes
    heuristic_indicators = torch.where(feasible_nodes, 
                                        -current_distance_matrix + noise,  # Lower scores for nearer nodes, higher for further
                                        torch.tensor(float('inf')).to(current_distance_matrix.device))  # Penalize infeasible nodes
    
    # Enhance exploration with weighted randomness
    heuristic_scores = heuristic_indicators + noise
    
    # Return the heuristic score matrix
    return heuristic_scores