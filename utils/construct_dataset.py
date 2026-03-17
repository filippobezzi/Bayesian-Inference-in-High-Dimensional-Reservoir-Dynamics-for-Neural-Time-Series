from utils.Reservoir import Reservoir
from utils.ESNDataset import ESNDataset
import torch

def construct_dataset(reservoir: Reservoir, time_series: torch.Tensor, external_mean = None, external_std = None, burnin=50, train_split=0.7, print_stats=False):
    """
    Standardizes input data, computes Reservoir states, and aligns them 
    with future targets for multi-region brain activity prediction.
    """
    L = len(time_series)
    reservoir.reset_state(batch_size=1)

    if (external_mean is None) or (external_std is None):
        # We use dim=0 to get a mean and std for each of the 119 regions
        mu = time_series.mean(dim=0)
        std = time_series.std(dim=0)
    else:
        mu = external_mean
        std = external_std
    # Avoid division by zero
    std_safe = std + 1e-8

    # This prevents reservoir explosion and keeps inputs in the optimal range
    time_series_scaled = (time_series - mu) / std_safe

    states = torch.zeros(L, reservoir.N)
    for t in range(L):
        # We use unsqueeze(0) to ensure the shape is [1, 119] for the reservoir
        sample = reservoir(time_series_scaled[t].unsqueeze(0))
        with torch.no_grad():
            states[t] = sample.squeeze(0)
    
    # We use state at time 't' to predict the scaled target at time 't+1'
    if burnin == 0:
        cleaned_states = states[:-1]
        cleaned_targets_scaled = time_series_scaled[1:]
    else:
        cleaned_states = states[burnin-1:-1]
        cleaned_targets_scaled = time_series_scaled[burnin:]

    # Use the length of the cleaned data for the split index
    num_samples = len(cleaned_states)
    split_index = int(num_samples * train_split)

    train_states = cleaned_states[:split_index]
    train_targets = cleaned_targets_scaled[:split_index]

    test_states = cleaned_states[split_index:]
    test_targets = cleaned_targets_scaled[split_index:]
    
    if print_stats:
        # Check that we have 119 distinct means/stds
        print(f"Stats computed for {mu.shape[0]} regions.")
        print(f"Train targets shape: {train_targets.shape}")

    # We pass the per-region mu and std so the plot function can 'un-scale' later
    train_data = ESNDataset(train_states, train_targets, mu, std)
    test_data = ESNDataset(test_states, test_targets, mu, std)   

    return train_data, test_data