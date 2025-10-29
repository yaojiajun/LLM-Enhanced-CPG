import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor,
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor,
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    max_distance = torch.max(current_distance_matrix)
    normalized_distance = current_distance_matrix / max_distance if max_distance > 0 else current_distance_matrix

    # Enhanced randomness through multiple layers of random weights
    rand_weights1 = torch.rand_like(normalized_distance) * 0.4 + 0.6  # [0.6, 1.0]
    rand_weights2 = torch.rand_like(normalized_distance) * 0.5  # [0.0, 0.5]

    # Dynamic score calculations integrating demands and constraints
    demand_penalty = torch.clamp(delivery_node_demands.unsqueeze(0) / current_load.unsqueeze(1), max=1)  # Penalty based on delivery demand
    time_window_score = torch.where((arrival_times <= time_windows[:, 1].unsqueeze(0)) & (arrival_times >= time_windows[:, 0].unsqueeze(0)),
                                     torch.ones_like(arrival_times), 
                                     torch.zeros_like(arrival_times))  # Binary score for time window feasibility

    score1 = (1 - normalized_distance) * rand_weights1 * demand_penalty  # Favor closer nodes with demand consideration
    score2 = normalized_distance * rand_weights2 - (1 - time_window_score)  # Penalize based on distance and time window constraints
    
    # Calculate heuristic scores
    heuristic_scores = score1 - score2

    # Ensure scores are normalized and bounded
    heuristic_scores = torch.clamp(heuristic_scores, min=-1, max=1)

    return heuristic_scores