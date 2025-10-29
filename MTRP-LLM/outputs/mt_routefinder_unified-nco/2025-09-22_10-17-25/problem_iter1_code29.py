import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the distance-based heuristic score matrix with randomness and distance normalization
    distance_heuristic = -current_distance_matrix / torch.mean(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Compute the demand-based heuristic score matrix with an emphasis on high-demand nodes and randomness
    delivery_score = delivery_node_demands.unsqueeze(0) * 0.5 - current_load.unsqueeze(1) * torch.max(
        delivery_node_demands) + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased noise for exploration
    noise_factor = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores with noise for exploration
    cvrp_scores = distance_heuristic + delivery_score + noise_factor

    # Rest of the code remains unchanged as per the original function

    return cvrp_scores