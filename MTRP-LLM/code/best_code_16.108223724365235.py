import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Generate a mutated distance-based heuristic score matrix with adjusted calculations
    mutated_distance_scores = -1 * (current_distance_matrix ** 1.3) * 0.3 - torch.randn_like(current_distance_matrix) * 0.2

    # Generate the demand-based heuristic score matrix without alteration but with a slight adjustment in coefficients
    modified_demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.2 + \
                             (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open))) * 0.4 + torch.randn_like(current_distance_matrix) * 0.2)

    # Calculate time-based heuristic score matrix with slightly updated time window constraints handling
    updated_time_score = ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.4) +
                          (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.6)).unsqueeze(0)

    # Calculate pickup-based heuristic score matrix with modified calculation for interaction with distance
    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix) ** 0.8) * 0.15
    pickup_score = torch.clamp(pickup_score, min=-float('inf'), max=float('inf'))  # Ensure finite values

    # Calculate the overall score matrix for edge selection using modified distance heuristic with refined scores integration
    overall_scores = mutated_distance_scores + modified_demand_scores - updated_time_score + pickup_score

    return overall_scores