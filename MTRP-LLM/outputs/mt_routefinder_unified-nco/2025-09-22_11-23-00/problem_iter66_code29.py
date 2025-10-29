import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    distance_heuristic = -1 * ((current_distance_matrix ** 1.3) - 1) * 0.2 + torch.randn_like(current_distance_matrix) * 0.3 # Modified distance heuristic

    delivery_score = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.1

    time_penalty = 0.4
    time_score = torch.min((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.3),
                           (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * time_penalty))

    adjusted_pickup_load = current_load.unsqueeze(1) + pickup_node_demands.unsqueeze(0).float()
    pickup_score = (adjusted_pickup_load / (1 + current_distance_matrix)).clamp(min=-1, max=1) * 0.16  # Balanced pickup_score
    
    open_delivery_score = -torch.sin(current_distance_matrix)*2
    
    route_duration_score = current_length.unsqueeze(1) - current_length.mean()

    overall_scores = distance_heuristic + delivery_score + time_score + pickup_score + open_delivery_score + route_duration_score
    
    return overall_scores