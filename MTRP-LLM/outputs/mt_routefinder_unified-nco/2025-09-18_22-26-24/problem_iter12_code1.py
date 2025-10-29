import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, 
                  current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, 
                  current_load_open: torch.Tensor, time_windows: torch.Tensor, 
                  arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, 
                  current_length: torch.Tensor) -> torch.Tensor:
    
    # Initialize the heuristic indicators
    base_scores = -current_distance_matrix.clone()  # Start with negative distances
    
    # Calculate urgency based on time windows: earlier windows could indicate higher urgency
    time_urgency = (time_windows[:, 0] - arrival_times) / (time_windows[:, 1] - time_windows[:, 0] + 1e-6)
    
    # Calculate load feasibility (whether vehicle can accommodate demands)
    load_feasibility = (current_load.unsqueeze(1) >= delivery_node_demands.unsqueeze(0)).float()
    load_feasibility_open = (current_load_open.unsqueeze(1) >= delivery_node_demands_open.unsqueeze(0)).float()
    
    # Calculate length feasibility
    length_feasibility = (current_length.unsqueeze(1) >= current_distance_matrix).float()
    
    # Combine all factors into the heuristic score
    heuristic_indicators = (
        base_scores + 
        time_urgency * 0.5 +
        load_feasibility * 0.2 +
        load_feasibility_open * 0.2 +
        length_feasibility * 0.3
    )
    
    # Introduce adaptive randomness
    adaptive_randomness = torch.rand_like(heuristic_indicators) * 0.1
    heuristic_indicators += adaptive_randomness
    
    # Ensure scores are within reasonable bounds
    heuristic_indicators = torch.clamp(heuristic_indicators, min=-1.0, max=1.0)

    return heuristic_indicators