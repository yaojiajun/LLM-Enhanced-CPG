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
    
    # Parameters
    num_nodes = delivery_node_demands.shape[0]
    pomo_size = current_load.shape[0]

    # Initialize heuristic score matrix
    heuristic_scores = torch.zeros((pomo_size, num_nodes), device=current_distance_matrix.device)

    # Check capacity feasibility
    delivery_feasible = delivery_node_demands.unsqueeze(0) <= current_load.unsqueeze(1)
    delivery_feasible_open = delivery_node_demands_open.unsqueeze(0) <= current_load_open.unsqueeze(1)

    # Time window checks
    within_time_window = (arrival_times < time_windows[:, 1].unsqueeze(0)) & (arrival_times > time_windows[:, 0].unsqueeze(0))
    
    # Compute penalties for infeasibility
    load_penalty = (~delivery_feasible) * -1e6  # Large negative value for infeasible delivery
    load_penalty_open = (~delivery_feasible_open) * -1e6  # Large negative value for infeasible open delivery
    
    # Update heuristic scores for delivery nodes that are feasible
    heuristic_scores += (delivery_feasible.float() * within_time_window.float() * (-current_distance_matrix + 1)).clamp(min=0)
    
    # Apply load penalty for infeasible deliveries
    heuristic_scores += load_penalty
    heuristic_scores += load_penalty_open

    # Factor in remaining duration
    duration_feasible = current_length.unsqueeze(1) >= current_distance_matrix
    duration_penalty = (~duration_feasible) * -1e6  # Large negative value for infeasible duration

    # Update scores based on duration feasibility
    heuristic_scores += duration_penalty

    # Introduce randomness to scores to escape local optima
    randomness = torch.rand_like(heuristic_scores) * 0.1  # Random noise to avoid convergence
    heuristic_scores += randomness

    return heuristic_scores