import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Generate random weights for enhanced randomness
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)

    # Calculate normalized metrics
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-5)
    customer_balance = (delivery_node_demands - pickup_node_demands).clamp(min=0)
    normalized_customer_balance = customer_balance / (torch.max(customer_balance) + 1e-5)

    # Include capacity and time window feasibility criteria
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    time_window_feasibility = ((arrival_times + normalized_distance) >= time_windows[:, 0].unsqueeze(0)).float() * \
                               ((arrival_times + normalized_distance) <= time_windows[:, 1].unsqueeze(0)).float()

    # Combine scores with different metrics
    score1 = (torch.tanh(normalized_distance) * 
               torch.sigmoid(normalized_customer_balance) * 
               load_feasibility * 
               time_window_feasibility * 
               rand_weights1)

    score2 = (torch.relu(torch.exp(current_distance_matrix)) + 
              torch.relu(delivery_node_demands - pickup_node_demands) + 
              rand_weights2)

    heuristic_scores = score1 - score2

    return heuristic_scores