import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    distance_heuristic = torch.tanh(-0.3 * current_distance_matrix)
    
    delivery_capability = torch.where(current_load.unsqueeze(1) >= 1.8 * delivery_node_demands_open, 1.6, -1.6)

    pickup_score = torch.sigmoid(2.0 * pickup_node_demands) * (2.0 / (1.0 + 0.1 * pickup_node_demands))

    total_score = 0.7 * distance_heuristic + 0.3 * delivery_capability - 0.7 * pickup_score

    open_delivery_score = torch.pow(torch.sigmoid(2.8 * delivery_node_demands_open), 3.2) * (4.7 / (1.3 + 1.8 * delivery_node_demands_open))
    
    # Mutated length score calculation with modified penalty for length exceeding threshold
    length_proximity = 1.0 - (current_length.unsqueeze(1) / (2.4 + 2.0 * current_distance_matrix)).clamp(max=1.0)
    length_score = -torch.pow(length_proximity, 1.8) * (1.8 + 0.2 * length_proximity)  # Adjusted penalty for length exceeding threshold
    
    total_score += 0.9 * open_delivery_score + 1.3 * length_score
    
    early_time_bonus = torch.clamp(time_windows[:, 0].unsqueeze(0) - arrival_times, min=0) * 0.8
    late_time_penalty = torch.clamp(arrival_times - time_windows[:, 1].unsqueeze(0), min=0) * 2.0
    
    randomness = torch.randn_like(total_score) * 0.25
    
    time_score = -1.9 * late_time_penalty - 1.4 * early_time_bonus
    
    total_score += time_score + randomness
    
    return total_score