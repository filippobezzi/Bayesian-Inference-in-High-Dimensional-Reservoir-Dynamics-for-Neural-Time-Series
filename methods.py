import pyro
import torch
import numpy as np

def model(S, Y=None, beta_params=[0,10.], sigma_params=[0,10.]):

    rows, cols = S.shape

    S = torch.asarray(S, dtype = torch.float64)
    if Y is not None: Y = torch.asarray(Y, dtype = torch.float64).squeeze()

    beta = pyro.sample(
        "beta",
        pyro.distributions.Normal(
            torch.tensor(beta_params[0], dtype=torch.float64),
            torch.tensor(beta_params[1], dtype=torch.float64)
        ).expand([cols]).to_event(1)
    )
    sigma = pyro.sample(
        "sigma",
        pyro.distributions.Uniform(
            torch.tensor(sigma_params[0], dtype=torch.float64),
            torch.tensor(sigma_params[1], dtype=torch.float64)
        )
    )
    
    mu = torch.matmul(beta, S.mT)

    with pyro.plate("data", rows):
        pyro.sample(
            "obs",
            pyro.distributions.Normal(mu, sigma),
            obs=Y,
        )