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
    
    # Calculate remaining capacities for delivery and pickups
    remaining_capacity_delivery = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    remaining_capacity_pickup = current_load_open.unsqueeze(1) + pickup_node_demands.unsqueeze(0)
    
    # Evaluate time window feasibility
    earliest_arrival = arrival_times + current_distance_matrix
    within_time_window = (earliest_arrival >= time_windows[:, 0].unsqueeze(0)) & (earliest_arrival <= time_windows[:, 1].unsqueeze(0))
    
    # Calculate feasible edges based on capacity and time windows
    feasible_delivery = (remaining_capacity_delivery >= 0) & within_time_window
    feasible_pickup = (remaining_capacity_pickup >= 0) & within_time_window
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros(current_distance_matrix.shape, device=current_distance_matrix.device)
    
    # Add positive scores for feasible delivery edges
    heuristic_scores += feasible_delivery.float() * (1 / (current_distance_matrix + 1e-6))  # Encouraging smaller distances
    
    # Add scores for feasible pickups
    heuristic_scores += feasible_pickup.float() * (1 / (current_distance_matrix + 1e-6)) * 0.5  # Less weight for pick-ups
    
    # Apply randomness to avoid local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Adding noise to scores
    heuristic_scores += randomness
    
    # Negative scores for infeasible edges
    heuristic_scores[~(feasible_delivery | feasible_pickup)] = -1e6  # Highly undesirable
    
    return heuristic_scores