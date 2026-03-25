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
            obs = pyro.sample("obs", dist.Normal(mu, sigma**2).to_event(1), obs=y)
        return mu

"""
class SSVS(PyroModule):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        return

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        # Global shrinkage
        tau = pyro.sample("tau", dist.HalfCauchy(1.0))  # (1,)
        # Local shrinkage
        with pyro.plate("features_plate", self.in_features):
            lambdas = pyro.sample("lambdas", dist.HalfCauchy(1.0))  # (1,)
            # Regression coefficients with SSVS prior
            beta = pyro.sample(
                "beta", dist.Normal(0.0, lambdas**2 * tau**2)
            )  # (in_features,)

        # bias = pyro.sample("bias", dist.Normal(0.0, 10.0))  # (1,)
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))  # (1,)

        mu = torch.matmul(x, beta)  # + bias  # (1,)

        if y is not None:
            y = y.squeeze(-1)

        with pyro.plate("data_plate", mu.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma**2), obs=y)  # (num_samples,)
        return mu
"""
class SSVS(PyroModule):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features # e.g., 119 regions
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
                dist.HalfCauchy(torch.ones(self.out_features, device=device)).to_event(1)
            )
            
            # beta shape: [in_features, out_features]
            # (Fixed parameterization: dist.Normal uses std_dev, not variance)
            beta = pyro.sample(
                "beta", 
                dist.Normal(torch.zeros(self.out_features, device=device), lambdas * tau).to_event(1)
            )

        # 3. Noise standard deviation per region: shape [out_features]
        # This allows each of the 119 regions to learn its own unique noise level
        sigma = pyro.sample(
            "sigma", 
            dist.Uniform(
                torch.zeros(self.out_features, device=device), 
                torch.full((self.out_features,), 10.0, device=device)
            ).to_event(1)
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
