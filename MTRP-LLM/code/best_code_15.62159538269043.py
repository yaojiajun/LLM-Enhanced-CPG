import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    open_delivery_score = torch.sigmoid(1.2 * delivery_node_demands_open) * (1.5 / (1.0 + 1.5 * delivery_node_demands_open))
    
    total_distance_heuristic = torch.pow(current_distance_matrix + 1e-5, -0.4)
    delivery_capability = torch.where(current_load.unsqueeze(1) >= 1.5 * delivery_node_demands_open, 1.6, -1.6)
    delivery_score = delivery_capability * (1.1 / (1.0 + 0.6 * delivery_node_demands))
    
    pickup_score = torch.sigmoid(0.4 * pickup_node_demands) * torch.exp(-0.7 * pickup_node_demands)
    pickup_penalty = torch.where(current_length.unsqueeze(1) < 0.4 * pickup_node_demands, -1.3, 0.0)
    
    normalized_length = current_length.unsqueeze(1) / (1.0 + current_distance_matrix)
    length_score = -torch.pow(normalized_length, 1.2) * 0.6
    
    total_score = 0.7 * total_distance_heuristic + 0.3 * delivery_score - 1.0 * (pickup_score + pickup_penalty) + 0.8 * open_delivery_score + 1.2 * length_score
    
    time_penalty = torch.clamp((arrival_times - 0.7 * time_windows[:, 1].unsqueeze(0)), min=0) * 0.7
    time_score = -3.8 * torch.clamp((arrival_times - 0.8 * time_windows[:, 0].unsqueeze(0)), min=0)
    
    total_score += time_penalty + time_score
    
    randomness = torch.randn_like(total_score) * 0.3
    total_score += randomness
    
    return total_score