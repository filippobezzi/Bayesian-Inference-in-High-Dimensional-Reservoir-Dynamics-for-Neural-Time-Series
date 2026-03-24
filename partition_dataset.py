import torch
import numpy as np
from torch.utils.data import Subset
from utils.ESNDataset import ESNDataset

def partition_dataset(dataset_list: list, num_partitions:10, training_frac=0.7, cal_frac=0.2, test_frac=0.1):
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
        for i in range(num_partitions):
            indices = np.arange(start_step, start_step + step)
            batches.append(Subset(ds, indices))
            start_step += step
    
    n_total = len(batches)
    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices) # Rimescoliamo l'ordine dei blocchi
    
    n_train = int(n_total * training_frac)
    n_cal = int(n_total * cal_frac)
    
    train_idx = all_indices[:n_train]
    cal_idx = all_indices[n_train : n_train + n_cal]
    test_idx = all_indices[n_train + n_cal:]

    def assemble_esn_dataset(indices_list):
        if len(indices_list) == 0: 
            return None
        
        selected_batches = [batches[i] for i in indices_list]
        
        all_states = torch.cat([b.dataset.states[b.indices] for b in selected_batches])
        all_preds = torch.cat([b.dataset.predictions[b.indices] for b in selected_batches])
        
    
        mu_val = dataset_list[0].mean
        std_val = dataset_list[0].std
    
        return ESNDataset(all_states, all_preds, mu_val, std_val)

    train_dataset = assemble_esn_dataset(train_idx)
    cal_dataset = assemble_esn_dataset(cal_idx)
    test_dataset = assemble_esn_dataset(test_idx)

    return train_dataset, cal_dataset, test_dataset