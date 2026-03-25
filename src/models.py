import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from pyro.nn import PyroModule, PyroSample
from typing import Callable


class BayesianModel(PyroModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_R: float = 1.0,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_R = sigma_R
        self.scale = scale
        return

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        R = pyro.sample(
            "R",
            dist.Normal(
                torch.tensor(0.0, device=x.device),
                torch.tensor(self.sigma_R**2 / self.in_features, device=x.device),
            )
            .expand([self.in_features, self.out_features])
            .to_event(2),
        )
        mu = torch.matmul(x, R)
        sigma = pyro.sample(
            "sigma",
            dist.HalfNormal(torch.tensor(self.scale, device=x.device))
            .expand([self.out_features])
            .to_event(1),
        )
        with pyro.plate("data", mu.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
        return mu


class SSVS(PyroModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        return

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device

        # 1. Global shrinkage (Scalar)
        tau = pyro.sample("tau", dist.HalfCauchy(torch.tensor(1.0, device=device)))

        # 2. Local shrinkage & Regression coefficients
        with pyro.plate("features_plate", self.in_features):
            # Using .to_event(1) expands the sample into shape [in_features, out_features]
            lambdas = pyro.sample(
                "lambdas",
                dist.HalfCauchy(torch.tensor(1.0, device=device))
                .expand([self.out_features])
                .to_event(1),
            )

            # beta shape: [in_features, out_features]
            beta = pyro.sample(
                "beta",
                dist.Normal(
                    torch.tensor(0.0, device=device),
                    lambdas * tau,
                ).to_event(1),
            )

        # 3. Noise standard deviation per region: shape [out_features]
        sigma = pyro.sample(
            "sigma",
            dist.Uniform(
                torch.tensor(0.0, device=device),
                torch.tensor(10.0, device=device),
            )
            .expand([self.out_features])
            .to_event(1),
        )

        # Matrix multiplication: [Batch, in_features] @ [in_features, out_features] -> [Batch, out_features]
        mu = torch.matmul(x, beta)

        # 4. Likelihood
        with pyro.plate("data_plate", x.shape[0]):
            # .to_event(1) tells Pyro these 119 targets belong to the same multi-dimensional observation
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu


class MLP(nn.Module):
    def __init__(
        self, layer_dims: list[int], activation: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.activation = activation
        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(
                nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            )
            layers.append(self.activation)
        layers.append(
            nn.Linear(in_features=layer_dims[-2], out_features=layer_dims[-1])
        )
        self._model = nn.Sequential(*layers)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class BayesianNeuralNetwork(PyroModule):
    def __init__(
        self,
        layer_sizes: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor],
        sigma_bias: float = 1.0,
        sigma_weight: float = 1.0,
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

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.sigma_bias = sigma_bias
        self.sigma_weight = sigma_weight
        self.layers: PyroModule | None = None
        return

    def _init_layers(self, device: torch.device) -> None:
        # initialize all the layers with their respective sizes
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[i], self.layer_sizes[i + 1]).to(
                device
            )
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.layers = PyroModule[nn.ModuleList](layer_list).to(device)

        # initialize all the prior distributions on the layers
        for i, layer in enumerate(self.layers):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            layer.weight = PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device=device),
                    torch.tensor(self.sigma_weight / input_dim, device=device),
                )
                .expand([output_dim, input_dim])
                .to_event(2)
            )

            layer.bias = PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device=device),
                    torch.tensor(self.sigma_bias, device=device),
                )
                .expand([output_dim])
                .to_event(1)
            )
        return

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        if self.layers is None:
            self._init_layers(device)

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        mu = self.layers[-1](x)  # [n_samples, n_features]

        sigma = pyro.sample(
            "sigma",
            dist.HalfNormal(torch.tensor(1.0, device=device))
            .expand([mu.shape[1]])
            .to_event(1),
        )  # [n_features]
        with pyro.plate("data", mu.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu
