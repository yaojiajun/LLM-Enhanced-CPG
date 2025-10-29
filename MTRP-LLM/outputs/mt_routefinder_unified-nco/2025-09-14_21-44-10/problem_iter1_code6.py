import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, 
                  pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate feasibility based on load constraints
    feasible_deliveries = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) & (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0))
    
    # Calculate feasibility based on time windows
    time_window_feasibility = ((arrival_times.unsqueeze(1) + current_distance_matrix <= time_windows[:, 1].unsqueeze(0)) & 
                                (arrival_times.unsqueeze(1) + current_distance_matrix >= time_windows[:, 0].unsqueeze(0)))

    # Calculate feasibility based on remaining lengths
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix)

    # Combine feasibility constraints with logical AND
    feasibility = feasible_deliveries & time_window_feasibility & length_feasibility

    # Calculate base heuristic score inversely proportional to distance for feasible routes
    base_scores = torch.where(feasibility, 1 / (current_distance_matrix + 1e-5), torch.tensor(float('-inf')).to(current_distance_matrix.device))

    # Incorporate randomness to escape local optima (based on existing edges)
    randomness = torch.rand_like(base_scores) * 0.1  # Adjust randomness magnitude as necessary

    # Final heuristic scores by summing base scores with randomness
    heuristic_scores = base_scores + randomness

    return heuristic_scores