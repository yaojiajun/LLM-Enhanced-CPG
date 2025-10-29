import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Implement improved heuristics based on problem constraints
    load_penalty = torch.where(current_load < delivery_node_demands, torch.tensor(-1000.0), torch.tensor(0.0))  # Penalize nodes with exceeding load capacity
    length_penalty = torch.where(current_length < current_distance_matrix, torch.tensor(-1000.0), torch.tensor(0.0))  # Penalize nodes that exceed route length constraints
    time_penalty = torch.where((arrival_times > time_windows[:, 1]) | (arrival_times < time_windows[:, 0]), torch.tensor(-1000.0), torch.tensor(0.0))  # Penalize nodes violating time windows
    pickup_penalty = torch.where(current_load_open < pickup_node_demands, torch.tensor(-1000.0), torch.tensor(0.0))  # Penalize nodes for exceeding pickup capacity

    heuristic_scores = load_penalty + length_penalty + time_penalty + pickup_penalty
    return heuristic_scores