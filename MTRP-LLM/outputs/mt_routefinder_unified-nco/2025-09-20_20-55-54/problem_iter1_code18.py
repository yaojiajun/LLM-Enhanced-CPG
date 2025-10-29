import torch
import torch
import torch.nn.functional as F

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    # Randomly perturb the distance matrix
    perturbed_distance_matrix = current_distance_matrix + torch.rand_like(current_distance_matrix) * 0.1
    
    # Calculate load ratios for delivery capacity check
    load_ratios = current_load.unsqueeze(1) / (delivery_node_demands.unsqueeze(0) + 1e-8)

    # Calculate load ratios for pickup and open route feasibility check
    load_open_ratio = current_load_open.unsqueeze(1) / (pickup_node_demands.unsqueeze(0) + 1e-8)

    # Adjust load ratios by keeping only finite values
    load_ratios = torch.where(torch.isfinite(load_ratios), load_ratios, torch.zeros_like(load_ratios))
    load_open_ratio = torch.where(torch.isfinite(load_open_ratio), load_open_ratio, torch.zeros_like(load_open_ratio))

    # Apply softmax to load ratios to further guide edge selection based on remaining capacity
    load_ratios = F.softmax(load_ratios, dim=1)
    load_open_ratio = F.softmax(load_open_ratio, dim=1)

    # Combine computed ratios and perturbed distances to generate final heuristic scores
    heuristic_scores = perturbed_distance_matrix * load_ratios + load_open_ratio
    
    return heuristic_scores