import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic scores
    heuristic_scores = torch.zeros_like(current_distance_matrix)
    
    # Calculate the remaining capacity usage for deliveries and pickups
    capacity_usage_delivery = delivery_node_demands / current_load.unsqueeze(1)
    capacity_usage_pickup = pickup_node_demands / current_load_open.unsqueeze(1)
    
    # Penalize for exceeding capacity
    capacity_penalty_delivery = torch.where(capacity_usage_delivery > 1, -1e6, torch.zeros_like(capacity_usage_delivery))
    capacity_penalty_pickup = torch.where(capacity_usage_pickup > 1, -1e6, torch.zeros_like(capacity_usage_pickup))
    
    # Calculate waiting time based on time windows
    current_time = arrival_times + current_distance_matrix
    waiting_time = torch.maximum(torch.zeros_like(current_time), time_windows[:, 0].unsqueeze(0) - current_time)
    
    # Penalize for not meeting time windows
    time_window_penalty = torch.where(current_time > time_windows[:, 1].unsqueeze(0), -1e6, torch.zeros_like(current_time))
    
    # Combine heuristics in a weighted manner
    heuristic_scores = (1 - capacity_usage_delivery * 0.5 + 
                        1 - capacity_usage_pickup * 0.5 + 
                        1 - (waiting_time / 10) + 
                        capacity_penalty_delivery + 
                        capacity_penalty_pickup + 
                        time_window_penalty)

    # Introduce randomness to avoid local optima
    adapt_randomness = torch.rand_like(heuristic_scores) * (1.0 / (1 + torch.abs(current_load.unsqueeze(1) - delivery_node_demands)))
    heuristic_scores += adapt_randomness

    return heuristic_scores