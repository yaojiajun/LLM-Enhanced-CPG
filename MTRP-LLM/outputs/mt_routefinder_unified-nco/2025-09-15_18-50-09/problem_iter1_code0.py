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
    
    # Number of customers
    num_nodes = delivery_node_demands.shape[0]
    pomo_size = current_distance_matrix.shape[0]

    # Initialize heuristic score matrix with zeros
    heuristic_scores = torch.zeros((pomo_size, num_nodes))

    # Calculate remaining capacity for delivery (both depot and open routes)
    remaining_capacity_delivery = current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)
    remaining_capacity_open = current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)

    # Calculate time window feasibility
    feasible_time_windows = (arrival_times + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) & \
                            (arrival_times + current_distance_matrix >= time_windows[:, 0].unsqueeze(0))

    # Calculate total route time and length feasibility
    feasible_length = current_length.unsqueeze(1) >= current_distance_matrix

    # Compute heuristic scores based on feasibility and distance
    # Favor close nodes while penalizing infeasible options
    feasible_scores = (feasible_time_windows & feasible_length & remaining_capacity_delivery & remaining_capacity_open).float()
    
    # Adding a random perturbation to enhance randomness in score calculation
    random_scores = torch.rand((pomo_size, num_nodes)) * 0.1
    heuristic_scores += (current_distance_matrix * feasible_scores) + random_scores

    # Normalize and clip scores to avoid extreme values
    heuristic_scores = torch.clamp(heuristic_scores, min=-1.0, max=1.0)

    # Return the heuristic score matrix
    return heuristic_scores