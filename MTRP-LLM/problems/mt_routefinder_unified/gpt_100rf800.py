import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    open_score = torch.zeros_like(current_distance_matrix)

    open_demand_factor = (2.0 / (1 + delivery_node_demands_open)).clamp(min=0.001)
    open_capacity_excess = (current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)).clamp(min=0)
    feasible_open_capacity = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))

    open_demand_weight = 0.4
    open_capacity_weight = 0.3
    open_proximity_weight = 0.6
    open_delivery_score = open_demand_weight * open_demand_factor - 0.2 * open_capacity_excess + open_proximity_weight * torch.exp(-current_distance_matrix)

    total_distance = current_distance_matrix.sum(dim=1, keepdim=True)
    distance_heuristic = (1.0 / (1 + total_distance)).clamp(min=0.001)

    adjusted_distance_matrix = torch.sqrt(current_distance_matrix + 1e-6)  # Non-linear transformation to distance
    distance_score = 1.5 * distance_heuristic * torch.exp(-adjusted_distance_matrix)

    noise_capacity = 0.8 * torch.randn_like(open_delivery_score)
    noise_distance_heuristic = 0.1 * torch.randn_like(distance_heuristic)

    # Updated time penalty calculation to consider the sum of squared lateness
    lateness = (arrival_times - time_windows[:, 1].unsqueeze(0)).clamp(0)  # Modified time penalty calculation
    squared_lateness = lateness ** 2
    time_penalty = 0.18 * squared_lateness.sum(dim=0)
    time_score = 0.6 * torch.exp(-0.3 * time_penalty)  # Modified time_score calculation

    length_score = (current_length.unsqueeze(1) - adjusted_distance_matrix).clamp(min=0)
    length_weight = 0.25

    normalized_length_score = length_weight * (length_score / (1 + current_length.unsqueeze(1))).clamp(min=0.001)

    adjusted_pickup_weight = 4.0 * pickup_node_demands  # Adjusted weight for pickup demands
    feasible_pickup = (current_load.unsqueeze(1) >= adjusted_pickup_weight.unsqueeze(0))
    excess_pickup = (adjusted_pickup_weight.unsqueeze(0) - current_load.unsqueeze(1)).clamp(min=0)

    # Enhanced pickup_score calculation focusing on encouraging feasible pickups
    pickup_score = torch.where(
        feasible_pickup,
        2.0 * (1.0 - (excess_pickup / (current_load.unsqueeze(1) + 1e-6)).clamp(0, 1)),
        -5.0 * (excess_pickup.pow(1.5) + 1e-6).sqrt()
    )

    heuristic_scores = 0.3 * distance_score + 0.4 * (open_delivery_score + noise_capacity) + 0.1 * open_score + noise_distance_heuristic + normalized_length_score + 0.1 * time_score + 0.1 * pickup_score

    return heuristic_scores