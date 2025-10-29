import torch
def heuristics_v2(current_distance_matrix, delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, time_windows, arrival_times, pickup_node_demands, current_length):
    # Generate random permutation matrices to introduce diversity in node selection
    perm_matrix = torch.rand_like(current_distance_matrix)  # Random permutation matrix
    permuted_distance_matrix = current_distance_matrix * perm_matrix

    # Introduce noise to distance matrix for exploration
    noise = torch.randn_like(current_distance_matrix) * 0.05  # Adding Gaussian noise with a small scale factor
    noisy_distance_matrix = current_distance_matrix + noise

    # Combine the permuted and noisy distance matrices to improve exploration
    improved_distance_matrix = permuted_distance_matrix + noisy_distance_matrix

    return improved_distance_matrix