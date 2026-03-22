import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from pyro.nn import PyroModule, PyroSample
from typing import Callable


class BayesianNeuralNetwork(PyroModule):
    def __init__(
        self,
        layer_sizes: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor],
        sigma_bias: float = 1.0,
        sigma_weight: float = 1.0,
        device: torch.device | str = "cpu",
    ):
        """
        Args:
            layer_sizes (list[int]):
                List containing the size of each layer in the network.
                Keep in mind that the first item in the list is the dimentionality of the input
                and the last one is the dimentionality of the output.

            activation (Callable[[torch.Tensor], torch.Tensor]):
                Activation function to apply after each layer of the network, except the output one.
        """

        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must have at least 2 elements: the input size and the output size."
            )

        super().__init__()

        self.device = device
        self.activation = activation

        # initialize all the layers with their respective sizes
        layer_list = [
            PyroModule[nn.Linear](layer_sizes[i], layer_sizes[i + 1]).to(self.device)
            for i in range(len(layer_sizes) - 1)
        ]
        self.layers = PyroModule[nn.ModuleList](layer_list).to(self.device)

        # initialize all the prior distributions on the layers
        for i, layer in enumerate(self.layers):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            layer.weight = PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(sigma_weight**2 / input_dim, device=self.device),
                )
                .expand([output_dim, input_dim])
                .to_event(2)
            )

            layer.bias = PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(sigma_bias**2, device=self.device),
                )
                .expand([output_dim])
                .to_event(1)
            )

        return

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        mu = self.layers[-1](x)  # [n_samples, n_features]

        sigma = pyro.sample(
            "sigma",
            # dist.Gamma(1, 1).expand([mu.shape[1]]).to_event(1),  # 1
            dist.HalfNormal(torch.tensor(1.0, device=self.device))
            .expand([mu.shape[1]])
            .to_event(1),
        )  # [n_features]
        with pyro.plate("data", mu.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma**2).to_event(1), obs=y)

        return mu
