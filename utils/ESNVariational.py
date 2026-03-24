import torch
from torch.utils.data import DataLoader
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoGuide
from pyro.optim import PyroOptim
from pyro.nn import PyroModule
from tqdm import tqdm

from utils.Reservoir import Reservoir


class _ESNVariational:
    def __init__(
        self, N, K, pyro_model, pyro_guide, optimizer, spectral_radius=0.9, scale=0.1
    ):

        # constructor attributes
        self.reservoir = Reservoir(N, K, spectral_radius=spectral_radius)
        self.pyro_model = pyro_model
        self.pyro_guide = pyro_guide
        self.optimizer = optimizer
        self.scale = scale
        self.svi = SVI(
            self.pyro_model, self.pyro_guide, self.optimizer, loss=Trace_ELBO()
        )

        # storage for Metrics
        self.loss_history = []

    def train(self, train_loader, epochs=500):
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
    def predict(self, test_states, num_samples=1000):
        predictive_object = Predictive(
            self.pyro_model, guide=self.pyro_guide, num_samples=num_samples
        )
        with torch.no_grad():
            samples = predictive_object(test_states, self.scale, None)
        y_samples = samples["obs"]
        return y_samples.squeeze()


class ESNVariational:
    def __init__(
        self,
        reservoir: Reservoir,
        pyro_model: PyroModule,
        pyro_guide: AutoGuide,
        optimizer: PyroOptim,
        scale: float = 0.1,
    ):
        # constructor attributes
        self.reservoir = reservoir
        self.pyro_model = pyro_model
        self.pyro_guide = pyro_guide
        self.optimizer = optimizer
        self.scale = scale
        self.svi = SVI(
            self.pyro_model, self.pyro_guide, self.optimizer, loss=Trace_ELBO()
        )

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
            self.pyro_model, guide=self.pyro_guide, num_samples=num_samples
        )
        with torch.no_grad():
            samples = predictive_object(test_states, self.scale, None)
        y_samples = samples["obs"]
        return y_samples.squeeze()
