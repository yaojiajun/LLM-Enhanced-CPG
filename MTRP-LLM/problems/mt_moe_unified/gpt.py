import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    distance_scale = 0.005
    distance_heuristic = torch.relu(current_distance_matrix) * distance_scale  # Modified distance heuristic computation using ReLU activation function

    delivery_penalty_factor = 2.0
    delivery_score_modifier = torch.exp(-delivery_node_demands / 20.0)

    delivery_demand_scaled = (current_load_open.unsqueeze(1) + 1e-6) / (delivery_node_demands_open.unsqueeze(0) + 1e-6)
    delivery_satisfaction_score = delivery_penalty_factor * (1 - torch.sigmoid(delivery_demand_scaled))

    distance_penalty = torch.log1p(current_distance_matrix)
    distance_penalty_weighted = torch.clamp(distance_penalty / (distance_penalty.max(dim=1, keepdim=True)[0] + 1e-6), max=1.0)
    distance_penalty_adjusted = torch.exp(distance_penalty_weighted * 3.0)

    gaussian_noise = torch.normal(mean=0.0, std=0.5, size=current_distance_matrix.shape)
    uniform_noise = torch.randint(-5, 5, current_distance_matrix.size(), dtype=torch.float)

    exploration_scores = delivery_satisfaction_score * delivery_score_modifier - (distance_heuristic + distance_penalty_adjusted) + gaussian_noise + uniform_noise

    random_mask = torch.rand_like(current_distance_matrix) > 0.15
    exploration_scores = exploration_scores * random_mask

    total_load = current_load.unsqueeze(1).expand(-1, current_distance_matrix.shape[1])
    demand_excess_penalty = (delivery_node_demands.unsqueeze(0) - total_load).clamp_min(0) * 1e6
    final_scores = exploration_scores - demand_excess_penalty

    load_positive_mask = total_load >= (delivery_node_demands.unsqueeze(0) * 0.9)
    load_penalty = (1 / (total_load + 1e-6)) * (1 - load_positive_mask.float()) * 1e4
    final_scores = final_scores + load_penalty

    feasible_mask = load_positive_mask.float()
    cvrp_scores = final_scores * feasible_mask + torch.normal(mean=0, std=0.6, size=current_distance_matrix.shape) * (1 - feasible_mask)

    earliest_times = time_windows[:, 0].unsqueeze(0)
    latest_times = time_windows[:, 1].unsqueeze(0)

    waiting_times = torch.clamp(earliest_times - arrival_times, min=0)
    late_arrivals = torch.clamp(arrival_times - latest_times, min=0)

    time_scores = (waiting_times + late_arrivals) * 0.5

    criticality_weights = torch.where(late_arrivals > 0, 2.5, 0.5)
    time_compensation = criticality_weights * time_scores

    vehicle_capacity = 100.0
    non_zero_demands = pickup_node_demands > 0
    used_capacity = torch.sum(pickup_node_demands.unsqueeze(0) * non_zero_demands.float(), dim=1)
    remaining_capacity = vehicle_capacity - used_capacity.unsqueeze(1)

    vrpb_compensation = torch.where(
        non_zero_demands,
        (remaining_capacity - pickup_node_demands) * non_zero_demands.float(),
        torch.zeros_like(cvrp_scores)
    ).clamp(min=0)

    feasible_mask_duration = (current_length.unsqueeze(1) > 0)
    vrpl_compensation = torch.where(feasible_mask_duration, 1 / (current_length.unsqueeze(1) + 1e-6),
                                    torch.zeros_like(cvrp_scores))

    pomo_size, N = cvrp_scores.shape
    noise_scale_factors = (0.4 + 0.6 * torch.rand((pomo_size, 1))) * (1 + 0.3 * torch.randn((pomo_size, N)).clamp(-1, 1))

    modified_load_ratio = (current_load.unsqueeze(1) + 1e-6) / (delivery_node_demands_open.unsqueeze(0) + 1e-6)
    modified_demand_sensitivity = (delivery_node_demands_open / (modified_load_ratio.clamp(min=1e-6))) * (1 + 0.5 * torch.rand((pomo_size, N))).clamp(0.1, 3.0)

    overall_scores = cvrp_scores + time_compensation + vrpb_compensation + vrpl_compensation + (noise_scale_factors * modified_demand_sensitivity)

    return overall_scores
