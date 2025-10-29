import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Compute heuristic score matrix based on various heuristics
    load_score = torch.where(current_load.unsqueeze(1) >= delivery_node_demands, torch.tensor(1.0), torch.tensor(-1.0))
    load_open_score = torch.where(current_load_open.unsqueeze(1) >= delivery_node_demands_open, torch.tensor(1.0), torch.tensor(-1.0))
    
    # penalize late visits
    waiting_time = torch.clamp(arrival_times - time_windows[:, 1].unsqueeze(0), min=0)
    time_score = torch.where(waiting_time <= 0, torch.tensor(1.0), torch.tensor(-1.0))
    
    # prioritize pickups to maintain a balanced load
    pickup_score = torch.where(current_load.unsqueeze(1) >= pickup_node_demands, torch.tensor(1.0), torch.tensor(-1.0))
    
    # encourage shorter routes
    length_score = torch.where(current_length.unsqueeze(1) >= current_distance_matrix, torch.tensor(1.0), torch.tensor(-1.0))
    
    # Combine individual scores to get final heuristic score matrix
    heuristic_score = load_score + load_open_score + time_score + pickup_score + length_score
    
    return heuristic_score