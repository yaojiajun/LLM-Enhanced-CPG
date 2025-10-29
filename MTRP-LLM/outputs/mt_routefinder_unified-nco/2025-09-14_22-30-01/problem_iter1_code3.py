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
    
    # Constants
    num_nodes = current_distance_matrix.shape[1]
    pomos_size = current_load.shape[0]

    # Initialize scores matrix
    scores = torch.zeros((pomos_size, num_nodes), device=current_distance_matrix.device)

    # Check availability of services based on load, time windows, and route duration
    feasible = (current_load.unsqueeze(1) >= delivery_node_demands[None, :]) & \
               (current_load_open.unsqueeze(1) >= delivery_node_demands_open[None, :]) & \
               (current_length.unsqueeze(1) >= current_distance_matrix) & \
               (arrival_times <= time_windows[:, 1][None, :]) & \
               (arrival_times >= time_windows[:, 0][None, :])

    # Calculate base scores based on distances
    distance_scores = 1 / (current_distance_matrix + 1e-6)  # avoiding division by zero
    scores += feasible.float() * distance_scores

    # Incorporate time window penalties
    time_penalties = torch.where(arrival_times < time_windows[:, 0][None, :],
                                  (time_windows[:, 0][None, :] - arrival_times), 
                                  torch.tensor(0.0, device=current_distance_matrix.device))
    
    scores -= feasible.float() * (time_penalties * 0.1)  # penalize for early arrivals

    # Add randomness to enhance exploration
    randomness = torch.rand_like(scores) * 0.05  # small random values
    scores += randomness

    # Combine delivery and pickup evaluation
    delivery_penalty = (current_load.unsqueeze(1) + delivery_node_demands[None, :]) > current_load.max()
    pickup_penalty = (current_load_open.unsqueeze(1) + pickup_node_demands[None, :]) > current_load_open.max()

    scores -= delivery_penalty.float() * 0.5  # penalize infeasible deliveries
    scores -= pickup_penalty.float() * 0.5    # penalize infeasible pickups

    return scores