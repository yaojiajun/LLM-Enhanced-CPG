import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    pomo_size, N_plus_1 = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, N_plus_1), device=current_distance_matrix.device)

    # Compute feasibility masks based on current load and delivery demands
    feasible_deliveries = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    feasible_open_deliveries = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()

    # Time window feasibility checks
    current_time = arrival_times + current_distance_matrix
    within_time_windows = ((current_time >= time_windows[:, 0].unsqueeze(0)) & (current_time <= time_windows[:, 1].unsqueeze(0))).float()

    # Capacity feasibility for pickups
    feasible_pickups = (current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0) <= 1.0).float() # Assuming max capacity is 1.0 for simplicity

    # Remaining length and duration budget
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine all feasibility masks
    feasibility_mask = feasible_deliveries * feasible_open_deliveries * within_time_windows * feasible_pickups * length_feasibility

    # Score for feasible routes (lower distance, higher score)
    heuristic_scores += (1 - current_distance_matrix) * feasibility_mask

    # Introduce enhanced randomness to the scores
    random_noise = torch.rand(heuristic_scores.shape, device=current_distance_matrix.device) * feasibility_mask
    heuristic_scores += random_noise * 0.1  # Adjust the intensity of randomness

    # Penalize undesirable routes
    undesirable_mask = 1 - feasibility_mask
    heuristic_scores -= undesirable_mask * 10  # Arbitrarily penalize to discourage infeasible routes

    return heuristic_scores