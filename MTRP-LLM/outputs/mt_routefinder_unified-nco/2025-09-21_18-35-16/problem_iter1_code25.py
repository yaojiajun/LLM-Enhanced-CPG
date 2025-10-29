import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Distance-based heuristic score matrix with modified calculations for edge selection
    distance_heuristic = torch.exp(-(current_distance_matrix - torch.min(current_distance_matrix)) / (torch.max(current_distance_matrix) - torch.min(current_distance_matrix)))

    # Delivery-based score for nodes considering load constraints
    delivery_score = 1 / (1 + torch.exp(-(delivery_node_demands - current_load.unsqueeze(1)) / torch.max(delivery_node_demands)))

    # Combine the distance and delivery scores with balanced weights for edge selection
    total_score = 0.6 * distance_heuristic + 0.4 * delivery_score

    return total_score