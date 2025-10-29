import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, 
                  delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, 
                  time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    epsilon = 1e-8
    
    significativo_nodes = (delivery_node_demands > 0).float()
    pickable_nodes = (pickup_node_demands > 0).float()
    
    # Calculate admissibility for delivery constraints
    admissibility_delivery = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)) * significativo_nodes

    # Calculate time window constraints
    service_time = arrival_times + current_distance_matrix
    # Mask by time windows
    in_time_windows = ((service_time >= time_windows[:, 0].unsqueeze(0)) & 
                       (service_time <= time_windows[:, 1].unsqueeze(0)))

    # Calculate accumulation of constrained paths
    reserved_capacity = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)
    feasible_capacity = (reserved_capacity >= 0).float() * admissibility_delivery
    
    # Gauge route efficiency based on feasible movements only
    route_efficiency = 1.0 / (current_distance_matrix + epsilon)  # Inverse distance
    heuristic_scores = route_efficiency * (feasible_capacity * in_time_windows)

    # Randomness for exploration
    exploration_noise = torch.randn_like(heuristic_scores) * 0.1
    heuristic_scores += exploration_noise 
    
    # Clamping scores to be finite and zero if not feasible
    heuristic_scores = torch.clamp(heuristic_scores, min=0)  # Ensuring non-negativity
    heuristic_scores[~feasible_capacity.bool()] = float('-inf')  # Mark infeasible locations
    
    return heuristic_scores