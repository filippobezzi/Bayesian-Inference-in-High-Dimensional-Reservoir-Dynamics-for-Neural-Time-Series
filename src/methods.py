import torch
import torch.nn as nn
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoGuide
from pyro.nn import PyroModule
from pyro.optim import PyroOptim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ESNDataset
from src.reservoir import Reservoir

from typing import Callable


class ESNMCMC:
    def __init__(self, model: PyroModule) -> None:
        self.model = model
        self.posterior_samples = None
        return

    def run(self, dataset: ESNDataset, **kwargs) -> None:
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, **kwargs)
        mcmc.run(dataset.states, dataset.x)
        self.posterior_samples = mcmc.get_samples()
        return

    def predict(self, dataset: ESNDataset) -> torch.Tensor:
        predictive = Predictive(self.model, self.posterior_samples)
        y_pred = predictive(dataset.states)
        return y_pred["obs"]


class ESNVariational:
    def __init__(
        self,
        model: PyroModule,
        guide: AutoGuide,
        optimizer: PyroOptim,
        scale: float = 0.1,
    ):
        # constructor attributes
        self.model = model
        self.guide = guide
        self.optimizer = optimizer
        self.scale = scale
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())

        # storage for Metrics
        self.loss_history = []
        return

    def train(self, train_loader: DataLoader, epochs: int = 500):
        """
        Training using a PyTorch DataLoader for mini-batch SVI.
        train_loader: Should yield (batch_states, batch_targets)
        """
        pbar = tqdm(range(epochs), desc="Training Pyro ESN")

        # Determine total dataset size for averaging loss
        dataset_size = len(train_loader.dataset)

        for epoch in pbar:
            epoch_loss = 0.0

            # Iterate over mini-batches
            for batch_states, batch_targets in train_loader:
                # SVI.step updates parameters based on this specific batch
                # It returns the ELBO loss for the batch
                loss = self.svi.step(batch_states, self.scale, batch_targets)
                epoch_loss += loss

            # Normalize loss by the total number of samples in the dataset
            avg_epoch_loss = epoch_loss / dataset_size
            self.loss_history.append(avg_epoch_loss)

            # Update progress bar every epoch, print details every 100
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Total ELBO Loss: {avg_epoch_loss:.4f}")

            pbar.set_postfix({"Loss": f"{avg_epoch_loss:.4f}"})

        return self.loss_history

    # evaluation
    def predict(self, test_states: torch.Tensor, num_samples: int = 1000):
        predictive_object = Predictive(
            self.model, guide=self.guide, num_samples=num_samples
        )
        with torch.no_grad():
            samples = predictive_object(test_states, self.scale, None)
        y_samples = samples["obs"]
        return y_samples.squeeze()


class ESNQR:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        return

    def train(self, train_loader: DataLoader, epochs: int) -> list[float]:
        loss_history = []
        for epoch in range(epochs):
            loss = torch.zeros(1)
            self.model.train()
            self.optimizer.zero_grad()
            for batch_states, batch_x in train_loader:
                batch_pred = self.model(batch_states)
                loss += self.loss_fn(batch_x, batch_pred)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
        return loss_history
    
    def predict(self, test_set: ESNDataset) -> torch.Tensor:
        self.model.eval()
        return self.model(test_set.states)
