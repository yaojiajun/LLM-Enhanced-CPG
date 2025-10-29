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
    
    pomo_size, N_plus_1 = current_distance_matrix.shape
    
    # Constraints checks
    load_check = delivery_node_demands.view(1, -1).expand(pomo_size, -1) + delivery_node_demands_open.view(1, -1).expand(pomo_size, -1) <= current_load.view(-1, 1).expand(-1, N_plus_1)
    time_check = ((arrival_times + current_distance_matrix >= time_windows[:, 0].view(1, -1).expand(pomo_size, -1)) & 
                   (arrival_times + current_distance_matrix <= time_windows[:, 1].view(1, -1).expand(pomo_size, -1)))
    
    length_check = current_length.view(-1, 1).expand(-1, N_plus_1) >= current_distance_matrix
    
    valid_moves = load_check & time_check & length_check

    # Compute a heuristic score using inverse distance where valid
    heuristic_scores = torch.where(valid_moves, 
                                    1 / (current_distance_matrix + 1e-6),  # Adding a small epsilon to avoid division by zero
                                    torch.tensor(float('-inf')).expand_as(current_distance_matrix))
    
    # Add randomness to encourage exploration
    random_factor = (torch.rand(pomo_size, N_plus_1) - 0.5) * 0.1  # Small perturbation
    heuristic_scores += random_factor
    
    # Summarizing load, time windows, and distance into a final heuristic score
    final_scores = torch.nan_to_num(heuristic_scores, nan=float('-inf'))
    
    # Prioritize nearer valid moves
    return final_scores