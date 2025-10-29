import torch
def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:

    # cvrp_scores
    # Balance distance-based heuristic score with increased focus on direct applicability
    distance_heuristic = -current_distance_matrix / (torch.max(current_distance_matrix) + 1e-5) + torch.randn_like(current_distance_matrix) * 0.5

    # Modify the delivery score to give more weight to nodes with high demand, enhancing overall score calculation
    delivery_score = ((delivery_node_demands.unsqueeze(0) - current_load.unsqueeze(1)) / (current_load.unsqueeze(1) + 1e-5)) * 1.5 + \
                     torch.max(delivery_node_demands) / 2 + torch.randn_like(current_distance_matrix) * 0.3

    # Introduce controlled randomness to enhance exploration potential 
    enhanced_noise = torch.randn_like(current_distance_matrix) * 1.5

    # Combine the modified heuristic scores considering updated strategies for more effective exploration
    cvrp_scores = distance_heuristic + delivery_score + enhanced_noise

    # Ensure that the outputs are finite
    cvrp_scores = torch.where(torch.isfinite(cvrp_scores), cvrp_scores, torch.zeros_like(cvrp_scores))

    return cvrp_scores