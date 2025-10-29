import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    mutated_distance_scores = -1 * (current_distance_matrix ** 1.7) * 0.5 - torch.randn_like(current_distance_matrix) * 0.2

    modified_delivery_scores = (current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0).float()) * 0.4 + \
                               (1 / (1 + torch.exp(torch.mean(current_load_open) - torch.min(current_load_open)).sqrt()) * 0.5 + torch.randn_like(current_distance_matrix) * 0.1)

    time_score = 0.5 * ((torch.max(arrival_times - time_windows[:, 0].unsqueeze(0), dim=0).values * 0.3) +  
                        (torch.max(time_windows[:, 1].unsqueeze(0) - arrival_times, dim=0).values * 0.7))

    adjusted_pickup_load = current_load.unsqueeze(1) + 2 * pickup_node_demands.unsqueeze(0).float()
    pickup_score = ((torch.exp(adjusted_pickup_load / 100) / (1 + current_distance_matrix) ** 1.5) + torch.randn_like(current_distance_matrix) * 0.1) * 0.2  

    pickup_score = torch.clamp(pickup_score, min=-float('inf'), max=float('inf'))

    open_delivery_score = (-current_distance_matrix ** 1.5 / (delivery_node_demands_open.unsqueeze(0) + torch.mean(current_load_open))).clamp(min=-1.5, max=2.5) * 0.3

    length_score = (current_length.unsqueeze(1) / (1 + current_distance_matrix)).clamp(min=-1.5, max=1.5) * 0.2  

    overall_scores = mutated_distance_scores + modified_delivery_scores - time_score + pickup_score + open_delivery_score + length_score

    return overall_scores