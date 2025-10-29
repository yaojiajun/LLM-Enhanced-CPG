import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Generate random weights to enhance exploration
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)
    rand_weights3 = torch.rand_like(current_distance_matrix)

    # Problem-specific metrics
    net_demand = delivery_node_demands - pickup_node_demands
    normalized_net_demand = net_demand / (torch.max(net_demand) + 1e-5)
    
    # Normalize distance considering maximum bounds
    normalized_distance = current_distance_matrix / (torch.max(current_distance_matrix) + 1e-5)
    
    # Time window feasibility
    time_window_feasibility = ((arrival_times + normalized_distance < time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times + normalized_distance > time_windows[:, 0].unsqueeze(0))).float()
    
    # Calculate heuristic scores
    score1 = (torch.tanh(normalized_distance) * torch.sigmoid(normalized_net_demand) +
              time_window_feasibility) * rand_weights1

    score2 = (torch.relu(torch.exp(current_distance_matrix)) +
              torch.relu(delivery_node_demands - pickup_node_demands) +
              torch.relu(current_load.unsqueeze(1) - delivery_node_demands) +
              torch.relu(current_length.unsqueeze(1) - current_distance_matrix) +
              rand_weights2 +
              rand_weights3)

    # Final score calculation
    heuristic_scores = score1 - score2

    return heuristic_scores