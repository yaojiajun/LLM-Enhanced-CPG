import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, 
                  delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize variables
    N = current_distance_matrix.shape[1] - 1  # Number of nodes
    pomo_size = current_distance_matrix.shape[0]  # Number of trajectories
    infinity = float('inf')

    # Availability conditions
    load_check = (current_load.unsqueeze(-1) >= delivery_node_demands.unsqueeze(0)).float()
    load_check_open = (current_load_open.unsqueeze(-1) >= delivery_node_demands_open.unsqueeze(0)).float()
    length_check = (current_length.unsqueeze(-1) >= current_distance_matrix).float()

    # Time window checks
    earliest_arrivals = arrival_times + current_distance_matrix
    time_window_check = ((earliest_arrivals >= time_windows[:, 0].unsqueeze(0)) & 
                         (earliest_arrivals <= time_windows[:, 1].unsqueeze(0))).float()

    # Combine checks to find valid edges
    valid_edges = load_check * load_check_open * length_check * time_window_check

    # Calculate heuristic scores
    scoring = (1 - valid_edges) * -infinity  # Penalize invalid edges
    scores = current_distance_matrix * valid_edges  # Positive scores for valid edges

    # Introduce enhanced randomness
    random_factor = torch.rand_like(scores) * 0.1
    scores += random_factor * valid_edges

    # Replace scores with negative values for invalid edges
    scores[scores == 0] = -infinity

    return scores