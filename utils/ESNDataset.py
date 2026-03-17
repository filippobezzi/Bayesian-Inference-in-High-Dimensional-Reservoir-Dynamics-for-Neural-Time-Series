import torch

class ESNDataset(torch.utils.data.Dataset):
    def __init__(self, states, predictions, mu, std):
          
        self.predictions = predictions
        self.states = states
        self.mean = mu
        self.std = std 
    
    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, index):
        return self.states[index], self.predictions[index]