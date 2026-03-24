import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from typing import Callable


class Reservoir(nn.Module):
    def __init__(
        self,
        num_neurons: int,
        input_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
        spectral_radius: float = 0.9,
    ) -> None:
        super(Reservoir, self).__init__()

        self.num_neurons = num_neurons  # Number of reservoir neurons
        self.input_features = input_features  # Input dimension

        # Initialize Input Matrix (Uniform distribution is common)
        W_in = (
            torch.rand(self.num_neurons, self.input_features) * 2 - 1
        )  # Range [-1, 1]

        # Initialize Reservoir Matrix (Gaussian)
        W = torch.randn(self.num_neurons, self.num_neurons)

        # Scale Spectral Radius for stability
        # We calculate the largest eigenvalue and scale W
        with torch.no_grad():
            # Use real/imaginary parts to find the magnitude of eigenvalues
            eigenvalues = torch.linalg.eigvals(W)
            max_eig = torch.max(torch.abs(eigenvalues))
            W = W * (spectral_radius / max_eig)

        # Register as buffers (Fixed weights, move with model to GPU)
        self.register_buffer("W_in", W_in)
        self.register_buffer("W", W)

        # Internal state (Initialized to zeros)
        self.register_buffer("state", torch.zeros(1, self.num_neurons))
        self._saved_states = {"default": torch.clone(self.state).detach()}

        self.activation = activation

        # eventual PCA module, to be added later if needed
        self.pca: PCA | None = None
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): value of the time serie at time $t$: $x_t$

        Returns:
            torch.Tensor: updated state: $s_{t+1} = f(W \cdot s_{t} + W_{in} \cdot x_{t})$.
                If `self.pca` is not `None`, then $s_{t+1}$ is first transformed by it.
        """

        # Linear combinations: Input effect + Reservoir recurrent effect
        # We use .t() on weights because x is [1, K] and states is [N, K]
        input_part = torch.matmul(x, self.W_in.t())
        recurrent_part = torch.matmul(self.state, self.W.t())

        # We .detach() to ensure we don't track gradients through time steps
        self.state = self.activation(input_part + recurrent_part).detach()

        return (
            torch.Tensor(self.pca.transform(self.state.numpy()))
            if self.pca is not None
            else self.state
        )

    def add_pca_module(self, pca: PCA) -> None:
        """
        From now on, all outputs of `forward` will be transformed through `pca` (already fitted).
        The internal update of `state` is unchanged.
        """
        self.pca = pca
        return

    def reset_state(self, batch_size=1) -> None:
        """Clears the memory of the reservoir."""
        self.state = torch.zeros(batch_size, self.num_neurons, device=self.state.device)
        return

    def save_state(self, tag: str = "default") -> None:
        """
        Saves the current state (named `tag`) in order to load it later.
        """
        self._saved_states[tag] = torch.clone(self.state).detach()
        return

    def load_state(self, tag: str = "default") -> None:
        """
        Loads the saved state named `tag`.
        """
        self.state = torch.clone(self._saved_states[tag]).detach()
        return
