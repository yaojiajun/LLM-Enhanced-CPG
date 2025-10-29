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
    
    # Initialize the heuristic score matrix with zeros
    pomo_size, N_plus_1 = current_distance_matrix.shape
    heuristic_scores = torch.zeros((pomo_size, N_plus_1), device=current_distance_matrix.device)

    # Calculate feasibility flags based on capacity, time windows, and current length
    delivery_capacity = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    delivery_open_capacity = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    time_window_feasibility = ((arrival_times.unsqueeze(1) >= time_windows[:, 0].unsqueeze(0)) &
                                (arrival_times.unsqueeze(1) <= time_windows[:, 1].unsqueeze(0))).float()
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()

    # Combined feasibility mask
    feasibility_mask = delivery_capacity * delivery_open_capacity * time_window_feasibility * length_feasibility

    # Compute base heuristic scores as inverse of distance for feasible routes
    heuristic_base = torch.where(feasibility_mask > 0, 1.0 / (current_distance_matrix + 1e-6), torch.tensor(0.0, device=current_distance_matrix.device))

    # Enhance randomness to avoid local optima
    randomness = torch.rand_like(heuristic_base) * 0.1  # small random noise
    heuristic_scores = heuristic_base + randomness

    # Scale scores to ensure positive contributions
    heuristic_scores = heuristic_scores * feasibility_mask

    return heuristic_scores