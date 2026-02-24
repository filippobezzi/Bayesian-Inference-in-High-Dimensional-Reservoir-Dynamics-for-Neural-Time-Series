import torch

class ESNDataset(torch.utils.data.Dataset):
    def __init__(self, states, predictions, mean=None, std=None):
        
        if mean is None or std is None:
            self.mean = predictions.mean()
            self.std = predictions.std()
        else:
            self.mean = mean
            self.std = std
            
        self.normalized_predictions = (predictions - self.mean) / self.std
        self.states = states 
    
    def __len__(self):
        return len(self.normalized_predictions)

    def __getitem__(self, index):
        return self.states[index], self.normalized_predictions[index]