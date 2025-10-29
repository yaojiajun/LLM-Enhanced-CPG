import torch
import torch

def heuristics_v2(current_distance_matrix: torch.Tensor, delivery_node_demands: torch.Tensor, current_load: torch.Tensor, delivery_node_demands_open: torch.Tensor, current_load_open: torch.Tensor, time_windows: torch.Tensor, arrival_times: torch.Tensor, pickup_node_demands: torch.Tensor, current_length: torch.Tensor) -> torch.Tensor:
    
    # Compute distance heuristic: Adjusted scaling for refined balance and emphasis on shorter distances
    distance_heuristic = -current_distance_matrix * 1.2  # Increased multiplier to slightly emphasize distance while maintaining focus
    
    # Compute delivery score based on remaining load with graded scoring for better constraint handling
    remaining_load = current_load.unsqueeze(1) - delivery_node_demands.unsqueeze(0)  # Shape: (pomo_size, N+1)
    delivery_score = torch.where(remaining_load > 0, remaining_load * 1.5, remaining_load * 0.3)  # Graded: higher reward for feasible, milder penalty for infeasible
    
    # Compute open delivery score based on remaining load for open routes with graded penalties
    open_remaining_load = current_load_open.unsqueeze(1) - delivery_node_demands_open.unsqueeze(0)  # Shape: (pomo_size, N+1)
    open_delivery_score = torch.where(open_remaining_load > 0, open_remaining_load * 1.4, open_remaining_load * 0.4)  # Strengthened penalty for infeasible: reward feasible, stronger penalty for infeasible
    
    # Compute time-related score based on time_windows and arrival_times with graded penalties
    earliest = time_windows[:, 0]  # Shape: (N+1)
    latest = time_windows[:, 1]    # Shape: (N+1)
    
    deviation_hi = arrival_times - latest[None, :]  # Shape: (pomo_size, N+1) - late arrivals
    deviation_lo = arrival_times - earliest[None, :]  # Shape: (pomo_size, N+1) - early arrivals
    
    # Graded penalties: quadratic for late, adjusted nonlinear for early
    late_penalty = torch.clamp(deviation_hi, min=0) ** 2 * 1.2  # Stronger graded penalty for lateness
    early_penalty = torch.clamp(-deviation_lo, min=0) ** 1.2 * 0.15  # Milder adjusted nonlinear penalty for earliness
    time_score = - (late_penalty + early_penalty)  # Negative score for penalties
    
    # Add dynamic noise based on the magnitude of deviations for controlled exploration
    deviation_magnitude = torch.abs(deviation_hi) + torch.abs(deviation_lo)
    noise_sigma = 0.05 + 0.06 * deviation_magnitude  # Dynamic noise scaling with adjusted coefficients
    time_score += torch.randn_like(time_score) * noise_sigma  # Integrate noise
    
    # Integrate time_score into the total score with adjusted weighting for better balance
    total_score = 0.15 * distance_heuristic + 0.6 * delivery_score + 0.3 * open_delivery_score + 0.6 * time_score  # Adjusted weight to 0.6 for prioritizing feasibility
    
    # Introduce controlled randomness for overall exploration
    randomness = torch.randn_like(total_score) * 0.3  # Adjusted scaling for better exploration balance
    final_score = total_score + randomness  # Final heuristic score matrix
    
    return final_score