import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    rand_weights1 = torch.rand_like(current_distance_matrix)
    rand_weights2 = torch.rand_like(current_distance_matrix)
    customer_metric = delivery_node_demands - pickup_node_demands

    normalized_distance = current_distance_matrix / torch.max(current_distance_matrix)
    normalized_customer_metric = customer_metric / torch.max(customer_metric)

    score1 = torch.tanh(normalized_distance) * torch.sigmoid(normalized_customer_metric) * rand_weights1
    score2 = torch.relu(torch.exp(current_distance_matrix)) + torch.relu(delivery_node_demands - pickup_node_demands) + rand_weights2

    heuristic_scores = score1 - score2

    return heuristic_scores