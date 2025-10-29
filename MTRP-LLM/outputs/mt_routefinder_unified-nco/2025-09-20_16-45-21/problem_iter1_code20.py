import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8
    num_nodes = current_distance_matrix.shape[1]
    num_trajectories = current_distance_matrix.shape[0]
    
    # Create a score matrix initialized to zero
    score_matrix = torch.zeros((num_trajectories, num_nodes), device=current_distance_matrix.device)

    # Analyze delivery feasibilities
    delivery_feasible = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Time window feasibility
    time_viability = ((arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)).float()) * \
                     ((arrival_times.unsqueeze(1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)).float())

    # Check length constraints
    length_viability = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combined feasibility
    feasibility = delivery_feasible * time_viability * length_viability
    
    # Cost scoring based on distance (inverse)
    cost_scores = 1 / (current_distance_matrix + epsilon)
    
    # Combining scores
    score_matrix = feasibility * cost_scores

    # Incorporate randomness to the scores to avoid premature convergence
    randomness = (torch.rand(num_trajectories, num_nodes, device=current_distance_matrix.device) * 0.01)
    score_matrix += randomness
    
    # Clamp scores to ensure they remain within finite bounds
    score_matrix = torch.clamp(score_matrix, min=0.0)

    return score_matrix