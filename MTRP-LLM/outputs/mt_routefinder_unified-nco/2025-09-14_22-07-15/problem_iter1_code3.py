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
    
    # Initialize heuristic score matrix
    scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate feasibility based on load constraints (for both delivery and pickups)
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) | \
                       (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Calculate feasibility based on time windows
    time_feasibility = (arrival_times < time_windows[:, 1].unsqueeze(0)) & \
                       (arrival_times + current_distance_matrix < time_windows[:, 0].unsqueeze(0))
    
    # Calculate feasibility based on current length constraints
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)
    
    # Combine feasibility evaluations
    feasible_edges = load_feasibility & time_feasibility & length_feasibility
    
    # Compute the heuristic scores based on distances for feasible edges
    scores[feasible_edges] = -current_distance_matrix[feasible_edges]  # Favor shorter distances
    
    # Add a randomness factor to enhance exploration
    randomness = torch.rand_like(scores) * (1 - feasible_edges.float())
    scores += randomness * 100  # Introduce higher weights where edges are infeasible
    
    # Normalize scores to have values in range [0, 1]
    min_scores = scores.min(dim=1, keepdim=True)[0]
    max_scores = scores.max(dim=1, keepdim=True)[0]
    normalized_scores = (scores - min_scores) / (max_scores - min_scores + 1e-6)  # Avoid division by zero

    return normalized_scores