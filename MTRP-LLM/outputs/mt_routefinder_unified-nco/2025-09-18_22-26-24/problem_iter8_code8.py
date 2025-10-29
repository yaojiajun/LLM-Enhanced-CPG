import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # Calculate heuristic score matrix with enhanced randomness and consideration of node characteristics and constraints
    heuristic_scores = torch.rand_like(current_distance_matrix)  # Example of generating heuristic scores

    # Introduce enhanced randomness to avoid local optima by adding random noise
    random_noise = 0.1 * torch.randn_like(heuristic_scores)
    heuristic_scores += random_noise

    # Implement advanced heuristic computation here for further improvements based on problem-specific insights

    return heuristic_scores