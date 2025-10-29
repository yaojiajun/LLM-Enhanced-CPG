import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Customized heuristics for model v5 with specific changes in distance, delivery, and load calculations

    # Adjust the calculation of distance score by normalizing current_distance_matrix and taking its square
    normalized_distance = (current_distance_matrix - torch.min(current_distance_matrix)) / (torch.max(current_distance_matrix) - torch.min(current_distance_matrix) + 1e-6)
    distance_heuristic = torch.square(normalized_distance)  # Updated distance heuristic using squared normalization

    # Modify the delivery score calculation by adding the squared difference with the mean delivery demand
    mean_delivery_demand = torch.mean(delivery_node_demands)
    delivery_score = torch.square(mean_delivery_demand - delivery_node_demands) * 0.4

    # Change how the load score is calculated by using the inverse of current_load
    load_score = 1 / (current_load + 1)  # Updated load heuristic using the inverse of load

    # Generate random noise matrix for additional exploration
    randomness_matrix = torch.randn_like(current_distance_matrix) * 0.2

    # Combine the heuristic scores with the updated calculations
    overall_scores = distance_heuristic + delivery_score.unsqueeze(0) + load_score.unsqueeze(1) + randomness_matrix

    return overall_scores