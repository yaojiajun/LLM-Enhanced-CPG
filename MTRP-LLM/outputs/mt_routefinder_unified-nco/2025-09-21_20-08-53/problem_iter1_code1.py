import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # cvrp_scores
    # Modify the normalized distance-based heuristic score matrix computation with different weights
    normalized_distance_scores = -current_distance_matrix / torch.max(current_distance_matrix) + torch.randn_like(
        current_distance_matrix) * 0.5

    # Modify the demand-based heuristic score matrix calculation with new adjustments
    demand_scores = (2 * delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.6 + 2 * torch.max(
        delivery_node_demands) / 3 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce increased randomness for exploration with higher noise level for improved diversity
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores with diversified strategies for balanced exploration
    cvrp_scores = normalized_distance_scores + demand_scores + enhanced_noise

    vrptw_scores
    # Extract the time window boundaries
    earliest_times = time_windows[:, 0].unsqueeze(0)  # Shape: (1, N+1)
    latest_times = time_windows[:, 1].unsqueeze(0)  # Shape: (1, N+1)

    # Calculate waiting times for early arrivals
    waiting_times = torch.clamp(earliest_times - arrival_times, min=0)

    # Calculate penalties for late arrivals
    late_arrivals = torch.clamp(arrival_times - latest_times, min=0)

    # Create adaptive weighting based on the criticality of nodes
    criticality_weights = torch.where(late_arrivals > 0, 1.5, 0.5)  # Penalty weight adjustment

    # Calculate time compensation with adaptive weighting
    time_compensation = criticality_weights * (waiting_times + late_arrivals)

    # Calculate final VRPTW scores without altering the original cvrp_scores
    vrptw_scores = time_compensation

    # vrpb_scores
    # Remaining implementation remains the same as the original function

    # vrpl_scores
    # Remaining implementation remains the same as the original function

    # ovrp_scores
    # Remaining implementation remains the same as the original function

    overall_scores = cvrp_scores + vrptw_scores + vrpb_scores + vrpl_scores + ovrp_scores

    return overall_scores