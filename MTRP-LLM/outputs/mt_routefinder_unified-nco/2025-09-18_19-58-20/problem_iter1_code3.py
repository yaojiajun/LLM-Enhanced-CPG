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

    # Initialize heuristic scores with zeros
    heuristic_scores = torch.zeros_like(current_distance_matrix)

    # Calculate demand satisfaction: lower @delivered demand is better
    delivery_satisfaction = delivery_node_demands.view(1, -1) <= current_load.view(-1, 1)
    heuristic_scores += delivery_satisfaction.float() * 1000  # Boost for feasible delivery

    # Calculate open route demands satisfaction
    delivery_satisfaction_open = delivery_node_demands_open.view(1, -1) <= current_load_open.view(-1, 1)
    heuristic_scores += delivery_satisfaction_open.float() * 1000  # Boost for feasible delivery in open route

    # Penalize over-demand scenarios
    over_demand_penalty = (delivery_node_demands.view(1, -1) > current_load.view(-1, 1)).float() * -1000
    heuristic_scores += over_demand_penalty
    
    # Evaluate time windows feasibility: early arrivals gain a score, late arrivals penalize
    current_time = arrival_times  # Assume we use current arrival estimations
    time_window_feasibility = (current_time >= time_windows[:, 0].view(1, -1)).float() * \
                               (current_time <= time_windows[:, 1].view(1, -1)).float()
    
    heuristic_scores += time_window_feasibility * 500  # positive scoring for within time windows

    # Introduce penalties for arrival that is too early concerning time windows
    too_early_penalty = (current_time < time_windows[:, 0].view(1, -1)).float() * -500
    heuristic_scores += too_early_penalty
    
    # Incorporate length duration limits: penalizing visits causing over-length paths
    length_penalty = (current_length.view(-1, 1) - current_distance_matrix < 0).float() * -1000
    heuristic_scores += length_penalty
    
    # Enhance randomness: introduce a small random component to each heuristic score
    randomness = torch.rand_like(heuristic_scores) * 5
    heuristic_scores += randomness
    
    return heuristic_scores