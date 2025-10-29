import torch
import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the inverse of the distance modified to emphasize nearby nodes more strongly
    epsilon = 1e-8
    distance_scores = 1 / (current_distance_matrix + epsilon)  # Inverse distance
    normalized_distance_scores = distance_scores / torch.max(distance_scores) - torch.randn_like(current_distance_matrix) * 0.5

    # Compute demand adjusting scores with a stronger penalty for exceeding load capacity
    demand_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)) * 0.6
    high_demand_penalty = torch.where(demand_scores < 0, -demand_scores, torch.zeros_like(demand_scores))
    demand_scores = demand_scores + high_demand_penalty + torch.randn_like(current_distance_matrix) * 0.3

    # Combine the distance and demand scores while introducing additional randomness
    cvrp_scores = normalized_distance_scores + demand_scores + torch.randn_like(current_distance_matrix) * 1.0

    # vrptw_scores
    # Extract the time window boundaries
    earliest_times = time_windows[:, 0].unsqueeze(0)  # Shape: (1, N+1)
    latest_times = time_windows[:, 1].unsqueeze(0)  # Shape: (1, N+1)

    # Calculate waiting times for early arrivals
    waiting_times = torch.clamp(earliest_times - arrival_times, min=0)

    # Calculate penalties for late arrivals
    late_arrivals = torch.clamp(arrival_times - latest_times, min=0)

    # Create adaptive weighting based on the criticality of nodes
    criticality_weights = torch.where(late_arrivals > 0, 1.7, 0.3)  # Penalty weight adjustment

    # Calculate time compensation with adaptive weighting
    time_compensation = criticality_weights * (waiting_times + late_arrivals)

    # Calculate final VRPTW scores without altering the original cvrp_scores
    vrptw_scores = time_compensation

    # vrpb_scores
    vehicle_capacity = 100.0

    # Intensive computations in vectorized form
    non_zero_demands = pickup_node_demands > 0

    # Calculate the used capacity for all pickups
    used_capacity = torch.sum(pickup_node_demands.unsqueeze(0) * non_zero_demands.float(), dim=1)

    # Calculate remaining capacity
    remaining_capacity = vehicle_capacity - used_capacity.unsqueeze(1)

    # Calculate backhaul compensation based on remaining capacity
    vrpb_compensation = torch.where(
        non_zero_demands,
        (remaining_capacity - pickup_node_demands) * non_zero_demands.float(),
        torch.zeros_like(cvrp_scores)
    )

    # Ensure compensation is clamped to be helpful, based on excess capacity
    vrpb_compensation = torch.clamp(vrpb_compensation, min=0)

    # Compute final vrpb_scores with broadcasting
    vrpb_scores = vrpb_compensation

    # vrpl_scores
    pomo_size, n_plus_1 = cvrp_scores.shape

    # Create a mask where the current_length is less than or equal to indices indicating feasible visits
    feasible_mask = (current_length.unsqueeze(1) > 0)  # Shape: (pomo_size, 1)

    # Compute duration criticality, amplifying scores for feasible paths
    vrpl_compensation = torch.where(feasible_mask, 1 / (current_length.unsqueeze(1) + 1e-6),
                                    torch.zeros_like(cvrp_scores))  # Avoid division by zero

    # Final computation of vrpl_scores
    vrpl_scores = vrpl_compensation


    # ovrp_scores
    pomo_size, N = cvrp_scores.shape

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Introduce multiple noise patterns with diverse scaling factors for broader exploration and randomness
    noise_scale_factors_1 = (0.5 + 0.5 * torch.rand((pomo_size, 1))) * (1 + 0.5 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.8 + 0.2 * torch.rand((pomo_size, N)))
    noise_scale_factors_2 = (0.6 + 0.4 * torch.rand((pomo_size, 1))) * (1 + 0.6 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.9 + 0.1 * torch.rand((pomo_size, N)))
    noise_scale_factors_3 = (0.4 + 0.6 * torch.rand((pomo_size, 1))) * (1 + 0.4 * torch.randn((pomo_size, N)).clamp(-1, 1)) * (0.7 + 0.3 * torch.rand((pomo_size, N)))

    # Dynamic demand sensitivity adjustment with broader scaling based on load and additional noise pattern
    demand_sensitivity = (delivery_node_demands_open / current_load_open.unsqueeze(1).clamp(min=1e-6)) * \
                         (1 + 0.5 * torch.rand((pomo_size, N))).clamp(0.1, 3.0)
    
    # Enhanced global variation for increased randomness combined with multiplied noise patterns
    global_variation = (0.5 + 0.5 * torch.rand((pomo_size, 1))) * \
                       (current_load_open.unsqueeze(1) * (torch.randn((pomo_size, N)).clamp(-1, 1)))

    # Adaptive scaling factor influenced by delivery demands for enhanced compensation
    adaptive_scale_factor = (0.8 + 0.2 * torch.rand((pomo_size, 1)))

    # Calculate the ovrp_compensation with diversified randomness and dynamic scaling
    ovrp_compensation = (noise_scale_factors_1 + noise_scale_factors_2 + noise_scale_factors_3 + global_variation) * \
                        (demand_sensitivity * adaptive_scale_factor)

    # Final ovrp_scores calculation
    ovrp_scores = ovrp_compensation
        
    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores