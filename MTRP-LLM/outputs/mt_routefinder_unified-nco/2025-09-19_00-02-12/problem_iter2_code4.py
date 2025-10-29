import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristic scores with a base score based on distances
    base_scores = -current_distance_matrix

    # Demand feasibility
    delivery_feasible = (current_load.unsqueeze(1) >= delivery_node_demands).float()
    pickup_feasible = (current_load_open.unsqueeze(1) >= pickup_node_demands).float()

    # Time window feasibility
    current_time = arrival_times + current_distance_matrix
    time_feasible = ((current_time >= time_windows[:, 0].unsqueeze(0)) & 
                     (current_time <= time_windows[:, 1].unsqueeze(0))).float()

    # Length feasibility
    length_feasible = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combine scores: higher score for feasible edges and lower scores for unfeasible
    heuristic_scores = (base_scores * delivery_feasible * pickup_feasible * time_feasible * length_feasible)

    # Introduce randomness to enhance exploration
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Small random value
    final_scores = heuristic_scores + randomness

    return final_scores