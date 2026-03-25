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
        """
        Args:
            model (PyroModule): model whose posterior distribution will be estimated with a MCMC.
        """
        self.model = model
        self.posterior_samples = None
        return

    def run(self, dataset: ESNDataset, **kwargs) -> None:
        """
        Run the MCMC to estimate the posterior distribution of `model`.

        Args:
            dataset (ESNDataset): training set.
            **kwargs (): additional keyword arguments that will be passed to `MCMC(kernel, **kwargs)`.
                Required arguments are `num_samples` and `warmup_steps`.
                Suggested arguments are `num_chains` and `mp_context`.
        """
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
        loss_fn: Trace_ELBO = Trace_ELBO(),
        scale: float = 0.1,
    ) -> None:
        # constructor attributes
        self.model = model
        self.guide = guide
        self.optimizer = optimizer
        self.scale = scale
        self.loss_fn = loss_fn
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.loss_fn)

        # storage for Metrics
        self.loss_history = []
        return

    def train(self, dataset: ESNDataset, epochs: int = 100, **kwargs) -> list[float]:
        """
        Args:
            dataset (ESNDataset): train set.
            epochs (int, optional): number of epochs. Defaults to 100.
            **kwargs (): additional keyword arguments that will be passed to `DataLoader(dataset, **kwargs)`.
                Suggested arguments are `batch_size` and `shuffle`.

        Returns:
            list[float]: loss value over the epochs.
        """
        train_loader = DataLoader(dataset, **kwargs)
        self.model.train()
        pbar = tqdm(range(epochs), desc="Training Pyro ESN")

        # Determine total dataset size for averaging loss
        dataset_size = len(dataset)

        for epoch in pbar:
            epoch_loss = 0.0

            # Iterate over mini-batches
            for batch_states, batch_targets in train_loader:
                # SVI.step updates parameters based on this specific batch
                # It returns the ELBO loss for the batch
                loss = self.svi.step(batch_states, batch_targets)
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
    def predict(self, dataset: ESNDataset, num_samples: int = 1000) -> torch.Tensor:
        self.model.eval()
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        with torch.no_grad():
            samples = predictive(dataset.states)
        y_samples = samples["obs"]
        return y_samples.squeeze()


class CheckLoss(nn.Module):
    def __init__(self, taus: list[float]) -> None:
        super().__init__()
        self.taus = taus
        return

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = predictions.device
        # If targets is a flat 1D array [Batch], force it to be 2D [Batch, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        batch_size, num_regions = targets.shape
        num_quantiles = len(self.taus)

        # Reshape the flat MLP output into [Batch, Quantiles, Regions]
        preds = predictions.view(batch_size, num_quantiles, num_regions)

        loss = torch.zeros(1, device=device)

        # Loop over each tau
        for i, tau in enumerate(self.taus):
            q_pred = preds[:, i, :]
            r = targets - q_pred

            quantile_loss = tau * torch.relu(r) + (1 - tau) * torch.relu(-r)
            loss += torch.mean(quantile_loss)

        return loss


class ESNQR:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: CheckLoss
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        return

    def train(self, dataset: ESNDataset, epochs: int = 100, **kwargs) -> list[float]:
        """
        Args:
            dataset (ESNDataset): train set.
            epochs (int, optional): number of epochs. Defaults to 100.
            **kwargs (): additional keyword arguments that will be passed to `DataLoader(dataset, **kwargs)`.
                Suggested arguments are `batch_size` and `shuffle`.

        Returns:
            list[float]: loss value over the epochs.
        """
        train_loader = DataLoader(dataset, **kwargs)
        device = dataset.states.device
        loss_history = []
        pbar = tqdm(range(epochs), desc="Training ESN")
        for epoch in pbar:
            loss = torch.zeros(1, device=device)
            self.model.train()
            self.optimizer.zero_grad()

            for batch_states, batch_x in train_loader:
                batch_pred = self.model(batch_states)

                loss += self.loss_fn(batch_pred, batch_x)

            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())

            # Update progress bar every epoch, print details every 100
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Total Loss: {loss.item():.4f}")

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return loss_history

    def predict(self, dataset: ESNDataset) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(dataset.states)

            # Automatically reshape for the calibration function!
            batch_size = raw_preds.shape[0]
            num_quantiles = len(self.loss_fn.taus)
            num_regions = raw_preds.shape[1] // num_quantiles

            # Final shape: [Batch, Quantiles, Regions]
            reshaped_preds = raw_preds.view(batch_size, num_quantiles, num_regions)

        return reshaped_preds
