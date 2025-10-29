import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Compute the normalized distance-based heuristic score matrix with added diversity through randomness
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.7

    # Compute the demand-based heuristic score matrix with emphasis on high-demand nodes and enhanced randomness
    demand_scores = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.8 + torch.max(
        delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.5

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 2.0

    # Combine the different heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

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

    # Create a mask where the current_length is less than or equal to feasible visits
    feasible_mask = (current_length.unsqueeze(1) > 0)  # Shape: (pomo_size, 1)

    # Compute duration criticality, amplifying scores for feasible paths
    vrpl_compensation = torch.where(feasible_mask, 1 / (current_length.unsqueeze(1) + 1e-6),
                                    torch.zeros_like(cvrp_scores))  # Avoid division by zero

    # Final computation of vrpl_scores
    vrpl_scores = vrpl_compensation
    
    # Adjustment to open delivery score calculation
    open_delivery_score = delivery_node_demands_open - current_load_open.unsqueeze(1)

    #op Constraints are modeled differently
    open_scores = torch.clamp(open_delivery_score, max=0) * torch.rand_like(open_delivery_score) * 1.3
    
    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores +  vrpl_scores + open_scores
    
    return overall_scores