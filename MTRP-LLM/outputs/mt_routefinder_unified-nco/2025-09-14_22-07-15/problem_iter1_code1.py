import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Combine different factors to compute heuristic scores
    demand_factor = delivery_node_demands.unsqueeze(0) / current_load.unsqueeze(-1)
    time_factor = (arrival_times + current_distance_matrix) / time_windows[:, 1].unsqueeze(0)
    pickup_factor = pickup_node_demands.unsqueeze(0) / current_load_open.unsqueeze(-1)
    length_factor = current_distance_matrix / current_length.unsqueeze(0)

    # Combine factors to compute final heuristic score matrix
    heuristic_scores = demand_factor + time_factor + pickup_factor + length_factor

    # Apply softmax for enhanced randomness and avoid convergence to local optima
    heuristic_scores = F.softmax(heuristic_scores, dim=1)

    return heuristic_scores