import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Compute heuristic scores based on various factors such as distance, load constraints, time windows, and route length limits
    distance_scores = F.relu(-current_distance_matrix)  # Higher distance means lower score, so negative distance multiplied by -1
    load_scores = F.relu(current_load - delivery_node_demands)  # Ensure remaining capacity is greater than or equal to delivery demand
    open_load_scores = F.relu(current_load_open - delivery_node_demands_open)  # Ensure open route capacity is sufficient
    time_window_scores = F.relu(arrival_times - time_windows[:, 1].unsqueeze(0)) + F.relu(time_windows[:, 0].unsqueeze(0) - arrival_times)  # Penalize early or late arrivals
    length_scores = F.relu(current_length)  # Ensure remaining route duration is within limits

    # Combine the scores using a weighted sum
    heuristic_scores = 0.2*distance_scores + 0.2*load_scores + 0.2*open_load_scores + 0.2*time_window_scores + 0.2*length_scores

    return heuristic_scores