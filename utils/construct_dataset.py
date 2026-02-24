from Reservoir import Reservoir
from ESNDataset import ESNDataset
import torch

def construct_dataset(reservoir: Reservoir, time_series, burnin = 50, train_split = 0.7, print_stats = False):
    """
    Function that takes as input a time series and gives a torch.utils.dataset object to 
    perform the training/testing
    """
    # Compute states for each input data
    L = len(time_series) 
    reservoir.reset_state(batch_size=1)

    states = torch.zeros(L,reservoir.N)
    for t in range(L):
        sample = reservoir(time_series[t])
        with torch.no_grad():
            states[t] = sample
    
    # Exclude burn_in to avoid dependence from initial input
    cleaned_states = states[burnin:-1]
    cleaned_time_series = time_series[burnin+1:]

    # Calculate split index
    split_index = int(L * train_split)

    # Divide training and test
    train_data_raw = cleaned_time_series[:split_index].view(-1)
    train_states_raw = cleaned_states[:split_index]

    test_data_raw = cleaned_time_series[split_index:].view(-1)
    test_states_raw = cleaned_states[split_index:]

    # compute mean and variance for training as control
    mu = train_data_raw.mean()
    sigma = train_data_raw.std()
    
    if print_stats:
        print(f"Mean targets = {mu:.4f}")
        print(f"Std targets = {sigma:.4f}")

    train_data = ESNDataset(train_states_raw, train_data_raw, mu, sigma)
    test_data = ESNDataset(test_states_raw, test_data_raw, mu, sigma)   

    return train_data, test_data 
