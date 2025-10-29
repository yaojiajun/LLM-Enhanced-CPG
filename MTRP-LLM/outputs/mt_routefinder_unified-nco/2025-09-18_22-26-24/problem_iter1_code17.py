import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Helper functions for heuristic calculations
    def calculate_load_balance_score(current_load, delivery_demands):
        return torch.abs(current_load - delivery_demands)

    def calculate_time_window_score(arrival_times, time_windows):
        return torch.max(torch.zeros_like(arrival_times), time_windows[:, 0] - arrival_times)

    def calculate_length_penalty(current_length):
        return F.relu(current_length)

    # Calculate heuristic scores based on various criteria
    load_balance_scores = calculate_load_balance_score(current_load, delivery_node_demands)
    time_window_scores = calculate_time_window_score(arrival_times, time_windows)
    length_penalties = calculate_length_penalty(current_length)

    # Combine scores into a final heuristic score matrix
    heuristic_scores = load_balance_scores + time_window_scores - length_penalties

    return heuristic_scores