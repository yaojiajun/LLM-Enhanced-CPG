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
    
    # Step 1: Generate randomized weights
    rand_weights = torch.rand_like(current_distance_matrix) * 0.1 + 0.9
    
    # Step 2: Normalization of the distance matrix
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-6)
    
    # Step 3: Compute scores for various heuristics
    time_penalty = (arrival_times - time_windows[:, 0].unsqueeze(0)).clamp(min=0) # Time window penalty
    capacity_penalty_delivery = (delivery_node_demands - current_load.unsqueeze(1)).clamp(min=0)  # Capacity constraints for deliveries
    capacity_penalty_pickup = (pickup_node_demands - current_load_open.unsqueeze(1)).clamp(min=0)  # Capacity constraints for pickups
    length_penalty = (current_length.unsqueeze(1) - current_distance_matrix).clamp(min=0)  # Route length limits

    # Step 4: Combine various components into a single score
    score1 = torch.tanh(normalized_distance) * rand_weights  # Distance factor
    score2 = torch.sigmoid(time_penalty) * torch.relu(capacity_penalty_delivery + capacity_penalty_pickup)  # Penalties for constraints
    score3 = (1 - torch.sigmoid(length_penalty)) * rand_weights  # Length constraint score

    heuristic_scores = score1 - score2 + score3  # Combine scores

    # Step 5: Normalize scores for consistency
    heuristic_scores = (heuristic_scores - heuristic_scores.min(dim=1, keepdim=True)[0]) / (heuristic_scores.max(dim=1, keepdim=True)[0] - heuristic_scores.min(dim=1, keepdim=True)[0] + 1e-6)

    return heuristic_scores