import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.reservoir import Reservoir


class ESNDataset(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.states = states.to(device)
        self.x = x.to(device)
        self.mu = mu
        self.std = std
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.states[index], self.x[index]

    def to(self, device: torch.device):
        return ESNDataset(
            states=self.states, x=self.x, mu=self.mu, std=self.std, device=device
        )


def construct_dataset(
    reservoir: Reservoir,
    time_series: torch.Tensor,
    external_mean: torch.Tensor | None = None,
    external_std: torch.Tensor | None = None,
    burnin: int = 50,
    train_split: float = 0.7,
    print_stats: bool = False,
) -> tuple[ESNDataset, ESNDataset]:
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

    states = torch.zeros(L, reservoir.num_neurons)
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
        cleaned_states = states[burnin - 1 : -1]
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


@dataclass
class DataSet:
    states: torch.Tensor
    x: torch.Tensor

    def to(self, device: torch.device) -> None:
        self.states = self.states.to(device)
        self.x = self.x.to(device)
        return


def partition_dataset(
    dataset_list: list[ESNDataset],
    num_partitions: int = 5,
    training_frac: float = 0.8,
    cal_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[ESNDataset, ESNDataset, ESNDataset]:
    """
    Function that, given a list of datasets, partitions each into a specified number of
    contiguous time-series sub-blocks. These blocks are then randomly shuffled and
    re-assembled into three distinct ESNDataset objects based on the provided
    percentage splits for training, calibration, and testing.

    INPUT:
    - dataset_list: List of ESNDataset objects (e.g., [train1, train2], where both train_split = 1.0).
    - num_partitions: Number of blocks to divide EACH dataset into.
    - training_frac: Fraction of total blocks assigned to the training set (default: 0.7).
    - cal_frac: Fraction of total blocks assigned to the calibration set (default: 0.2).
    - test_frac: Fraction of total blocks assigned to the test set (default: 0.1).

    OUTPUT:
    - train_dataset: A single ESNDataset containing the shuffled training blocks.
    - cal_dataset: A single ESNDataset containing the shuffled calibration blocks.
    - test_dataset: A single ESNDataset containing the shuffled test blocks.

    Note:
    Internal temporal order is preserved within each block, but the order of blocks
    within the final datasets is randomized to improve model generalization across
    different time windows/trials.
    """
    batches = []

    for ds in dataset_list:
        step = len(ds) // num_partitions
        start_step = 0
        for i in range(num_partitions - 1):
            indices = np.arange(start_step, start_step + step)
            batches.append(Subset(ds, indices))
            start_step += step
        indices = np.arange(start_step, ds.states.shape[0])
        batches.append(Subset(ds, indices))

    n_total = len(batches)
    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)  # Rimescoliamo l'ordine dei blocchi

    n_train = int(n_total * training_frac)
    n_cal = int(n_total * cal_frac)

    train_idx = all_indices[:n_train]
    cal_idx = all_indices[n_train : n_train + n_cal]
    test_idx = all_indices[n_train + n_cal :]

    def assemble_esn_dataset(indices_list):
        if len(indices_list) == 0:
            return None

        selected_batches = [batches[i] for i in indices_list]

        all_states = torch.cat([b.dataset.states[b.indices] for b in selected_batches])
        all_preds = torch.cat([b.dataset.x[b.indices] for b in selected_batches])

        mu_val = dataset_list[0].mu
        std_val = dataset_list[0].std

        return ESNDataset(all_states, all_preds, mu_val, std_val)

    train_dataset = assemble_esn_dataset(train_idx)
    cal_dataset = assemble_esn_dataset(cal_idx)
    test_dataset = assemble_esn_dataset(test_idx)

    return train_dataset, cal_dataset, test_dataset
