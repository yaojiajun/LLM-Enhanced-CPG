import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Define constants
    penalty_factor = 1000.0
    randomness_factor = torch.rand(current_distance_matrix.shape) * 0.1
    
    # Capacity constraints (penalty if demand exceeds remaining load)
    delivery_capacity_constraint = (delivery_node_demands[:, None] <= current_load[None, :]).float()
    pickup_capacity_constraint = (pickup_node_demands[:, None] <= current_load_open[None, :]).float()

    # Time window constraints
    time_window_start = time_windows[:, 0][None, :]
    time_window_end = time_windows[:, 1][None, :]
    
    arrival_time_constraints = (arrival_times + current_distance_matrix < time_window_end) & \
                               (arrival_times + current_distance_matrix > time_window_start)

    # Remaining route length constraint
    length_constraint = (current_length[:, None] >= current_distance_matrix).float()

    # Calculate heuristic scores
    feasible_roads = delivery_capacity_constraint * pickup_capacity_constraint * arrival_time_constraints * length_constraint
    
    # Compute base score from distance, modify with random exploration
    base_score = -current_distance_matrix * feasible_roads + randomness_factor
    
    # Applying penalties for infeasible edges
    penalty_score = (1 - feasible_roads) * penalty_factor
    
    # Final heuristic score matrix
    heuristic_scores = base_score - penalty_score
    
    return heuristic_scores