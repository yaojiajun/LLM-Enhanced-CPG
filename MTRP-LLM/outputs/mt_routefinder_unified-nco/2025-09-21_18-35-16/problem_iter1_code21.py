import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Redefine the normalized distance-based heuristic score matrix
    epsilon = 1e-8
    distance_adjustment = current_distance_matrix / (torch.max(current_distance_matrix, dim=1, keepdim=True)[0] + epsilon)
    normalized_distance_scores = -distance_adjustment + torch.randn_like(current_distance_matrix) * 0.5

    # Redefine the demand-based heuristic score matrix
    remaining_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    demand_scores = (remaining_capacity.clamp(min=0) * 0.9) + (torch.max(delivery_node_demands) / 3) + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the different heuristic scores
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    # vrptw_scores
    earliest_times = time_windows[:, 0].unsqueeze(0)  # Shape: (1, N+1)
    latest_times = time_windows[:, 1].unsqueeze(0)  # Shape: (1, N+1)

    waiting_times = torch.clamp(earliest_times - arrival_times, min=0)
    late_arrivals = torch.clamp(arrival_times - latest_times, min=0)
    criticality_weights = torch.where(late_arrivals > 0, 1.7, 0.3)

    time_compensation = criticality_weights * (waiting_times + late_arrivals)
    vrptw_scores = time_compensation

    # vrpb_scores
    vehicle_capacity = 100.0
    non_zero_demands = pickup_node_demands > 0
    used_capacity = torch.sum(pickup_node_demands.unsqueeze(0) * non_zero_demands.float(), dim=1)
    remaining_capacity = vehicle_capacity - used_capacity.unsqueeze(1)
    vrpb_compensation = torch.where(
        non_zero_demands,
        (remaining_capacity - pickup_node_demands) * non_zero_demands.float(),
        torch.zeros_like(cvrp_scores)
    )
    vrpb_compensation = torch.clamp(vrpb_compensation, min=0)
    vrpb_scores = vrpb_compensation

    # vrpl_scores
    pomo_size, n_plus_1 = cvrp_scores.shape
    feasible_mask = (current_length.unsqueeze(1) > 0)
    vrpl_compensation = torch.where(feasible_mask, 1 / (current_length.unsqueeze(1) + 1e-6),
                                    torch.zeros_like(cvrp_scores))
    vrpl_scores = vrpl_compensation

    # ovrp_scores
    pomo_size, N = cvrp_scores.shape

    torch.manual_seed(42)
    noise_scale_factors_1 = (0.5 + 0.5 * torch.rand((pomo_size, 1))) * (1 + 0.5 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.8 + 0.2 * torch.rand((pomo_size, N)))
    noise_scale_factors_2 = (0.6 + 0.4 * torch.rand((pomo_size, 1))) * (1 + 0.6 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.9 + 0.1 * torch.rand((pomo_size, N)))
    noise_scale_factors_3 = (0.4 + 0.6 * torch.rand((pomo_size, 1))) * (1 + 0.4 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.7 + 0.3 * torch.rand((pomo_size, N)))

    demand_sensitivity = (delivery_node_demands_open / current_load_open.unsqueeze(1).clamp(min=1e-6)) * (1 + 0.5 * torch.rand((pomo_size, N))).clamp(0.1, 3.0)
    global_variation = (0.5 + 0.5 * torch.rand((pomo_size, 1))) * (current_load_open.unsqueeze(1) * (torch.randn((pomo_size, N)).clamp(-1, 1)))

    adaptive_scale_factor = (0.8 + 0.2 * torch.rand((pomo_size, 1)))

    ovrp_compensation = (noise_scale_factors_1 + noise_scale_factors_2 + noise_scale_factors_3 + global_variation) * (demand_sensitivity * adaptive_scale_factor)

    ovrp_scores = ovrp_compensation
        
    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores