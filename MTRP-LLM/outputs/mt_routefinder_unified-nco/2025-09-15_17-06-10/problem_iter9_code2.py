import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Introduce random weights for diversification
    rand_weights1 = torch.rand_like(current_distance_matrix) * 0.5 + 0.5
    rand_weights2 = torch.rand_like(current_distance_matrix) * 0.5 + 0.5
    
    # Normalize distance matrix
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-10)

    # Calculate scores based on modified heuristics
    score_distance = torch.tanh(normalized_distance) * rand_weights1
    score_load = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float() * (1 - delivery_node_demands.unsqueeze(0) / 100.0)  # example threshold for demands
    score_time = (time_windows[:, 0] <= arrival_times) & (arrival_times <= time_windows[:, 1])
    score_time = score_time.float()

    # Calculate score based on pickups and length
    score_pickup = (current_load_open.unsqueeze(1) >= pickup_node_demands.unsqueeze(0)).float() * (1 - pickup_node_demands.unsqueeze(0) / 100.0)  # example threshold for pickups
    score_length = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine scores with enhanced randomness
    heuristic_scores = (score_distance + score_load + score_time + score_pickup + score_length * rand_weights2) / 5.0
    
    # Introduce a slight randomness to the final scores
    heuristic_scores += torch.randn_like(heuristic_scores) * 0.1  # Adding noise for exploration
    
    return heuristic_scores