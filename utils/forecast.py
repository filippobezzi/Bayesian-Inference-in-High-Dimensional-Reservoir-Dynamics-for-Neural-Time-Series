import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from pyro.infer import Predictive
from typing import Callable

from utils.Reservoir import Reservoir


def recursive_forecast(
    x: torch.Tensor,
    predictive: Predictive,
    reservoir: Reservoir,
    iterations: int = 10,
    n_chains: int = 100,
) -> torch.Tensor:
    if x.shape[0] > 1 or len(x.shape) != 2:
        raise ValueError(
            "x must be a Tensor of shape (1, n_features), with n_features > 0"
        )

    # internal function that generates a single chain
    def _generate_forecast_chain(
        x: torch.Tensor,
        predictive: Predictive,
        reservoir: Reservoir,
        iterations: int = 10,
    ) -> torch.Tensor:

        chain = torch.zeros(size=(iterations + 1, x.shape[1]))
        chain[0, :] = x
        for i in range(iterations):
            state = reservoir(chain[i, :])
            chain[i + 1, :] = torch.mean(predictive(state)["obs"], dim=0).reshape(-1)
        return chain

    # setting up the checkpoint state to reset to
    reservoir.save_state()
    chains = []
    for _ in range(n_chains):
        # reset state for every chain in order to have the same starting point
        reservoir.load_state()
        # need to add a third dimension for later concatenation
        chains.append(
            _generate_forecast_chain(x, predictive, reservoir, iterations)[None, :, :]
        )
    return torch.concat(chains, dim=0)


if __name__ == "__main__":
    pass
