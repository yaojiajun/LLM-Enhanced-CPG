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
    
    # Heuristic score calculation based on various factors
    capacity_penalty = (delivery_node_demands.unsqueeze(0) > current_load.unsqueeze(1)).float() * -1000
    capacity_penalty_open = (delivery_node_demands_open.unsqueeze(0) > current_load_open.unsqueeze(1)).float() * -1000
    
    # Time window penalties
    time_penalty = ((arrival_times.unsqueeze(1) < time_windows[:, 0].unsqueeze(0)) | 
                    (arrival_times.unsqueeze(1) > time_windows[:, 1].unsqueeze(0))).float() * -1000
    
    # Duration limits
    duration_penalty = (current_distance_matrix > current_length.unsqueeze(1)).float() * -1000
    
    # Compute heuristic scores: distance is inversely related to desirability, i.e., shorter distances are preferred
    distance_score = -current_distance_matrix
    
    # Combining penalties to form a final score matrix
    score_matrix = distance_score + capacity_penalty + capacity_penalty_open + time_penalty + duration_penalty
    
    # Introduce randomness to the scores for better exploration
    randomness = torch.randn_like(score_matrix) * 0.1  # small random value added
    score_matrix += randomness
    
    # Ensure scores are adjusted to avoid extremely negative values if needed
    score_matrix = torch.clamp(score_matrix, min=-1000, max=None)
    
    return score_matrix