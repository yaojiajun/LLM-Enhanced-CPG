import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    
    # Compute heuristic scores based on various factors
    score_matrix = torch.zeros_like(current_distance_matrix)
    
    # Example: Compute scores based on distance matrix and load constraints
    load_ratio = (current_load.unsqueeze(1) - delivery_node_demands) / (torch.max(current_load) + 1e-8)
    distance_score = torch.exp(-current_distance_matrix)  # Score inversely proportional to distance
    load_score = F.relu(load_ratio)  # Only positive load ratios are feasible
    
    # Introduce controlled randomness in scoring
    noise = torch.randn_like(distance_score) * 0.05  # Small noise to break ties
    
    # Aggregate different scores to form the final score matrix
    score_matrix = distance_score + load_score + noise
    
    return score_matrix