import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v5
    
    # Modify the calculation of distance score by incorporating a squared exponential transformation
    distance_heuristic = torch.exp(-torch.square(current_distance_matrix)) / (torch.max(current_distance_matrix) + 1e-6)  # Squared exponential distance heuristic

    # Change how the delivery score is calculated by introducing a penalty for high demands
    delivery_score = (delivery_node_demands - torch.mean(delivery_node_demands)) / (torch.max(delivery_node_demands) + 1e-6) * 0.5
    
    # Adjust the load score by normalizing based on the mean and penalizing extreme load values
    load_score = (current_load - torch.mean(current_load)) / (torch.std(current_load) + 1e-6) - (current_load.abs() - torch.mean(current_load).abs()) * 0.3

    # Compute a random matrix for exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine heuristic scores with updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores