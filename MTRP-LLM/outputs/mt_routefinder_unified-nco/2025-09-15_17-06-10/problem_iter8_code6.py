import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    rand_weights = torch.rand_like(current_distance_matrix) * 0.5  # Scale for effective random weighting
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-5)  # Adding small constant for numerical stability
    
    # Score based on distance while considering the current load and vehicle capacity
    load_constraint = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    
    # Time window feasibility evaluation
    time_feasibility = ((arrival_times + current_distance_matrix) >= time_windows[:, 0].unsqueeze(0)).float() * \
                       ((arrival_times + current_distance_matrix) <= time_windows[:, 1].unsqueeze(0)).float()
    
    # Heuristic score computation
    heuristic_scores = (load_constraint * time_feasibility) * (torch.sigmoid(normalized_distance) + rand_weights)
    
    return heuristic_scores