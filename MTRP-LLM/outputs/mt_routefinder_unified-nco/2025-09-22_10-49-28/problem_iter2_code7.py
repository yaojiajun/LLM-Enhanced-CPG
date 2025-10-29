import torch
import numpy as np
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # New computation for normalized distance-based heuristic score matrix with revised approach
    distance_heuristic = torch.log(current_distance_matrix + 1) / 10 - torch.rand_like(current_distance_matrix) * 0.5

    # New computation for demand-based heuristic score matrix with adjusted weights and noise levels
    delivery_score = (delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) * 0.4 + torch.max(current_load) * 0.6 + torch.rand_like(current_distance_matrix) * 0.3

    # Combine the updated heuristic scores with varied strategies
    heuristics_scores = distance_heuristic + delivery_score

    # Keep the calculations related to other inputs unchanged

    # Return the modified heuristic score matrix
    return heuristics_scores