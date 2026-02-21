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

def evaluate_metrics(y_pred_samples, y_true, coverage_prob=0.95, k_levels=[0.025, 0.5, 0.975]):
    # y_true shape = (test_set_size,)
    # y_pred_samples shape = (num_samples, test_set_size)
    y_true = torch.from_numpy(y_true)

    # 1. Empirical Coverage
    Y_lower = torch.quantile(y_pred_samples, (1-coverage_prob)/2, dim=0)
    Y_upper = torch.quantile(y_pred_samples, (1+coverage_prob)/2, dim=0)

    coverage_mask = (y_true >= Y_lower) & (y_true <= Y_upper)
    ecov = coverage_mask.float().mean().item()

    # 2. Calibration Error
    k_coverages = []
    for k in k_levels:
        quantile_k = torch.quantile(y_pred_samples, k, dim=0)
        k_coverage = (y_true <= quantile_k).float().mean().item()
        k_coverages.append(k_coverage)

    cal = sum((tau - tau_hat)**2
          for tau, tau_hat in zip(k_levels, k_coverages))

    # 3. mCRPS
    tau_grid = torch.linspace(0.0, 1.0, 101, dtype=y_pred_samples.dtype)

    q = torch.quantile(y_pred_samples, tau_grid, dim=0) # shape = (num_samples, test_set_size)

    T = y_true.shape[0]
    crps_vals = []

    for t in range(T):
        y_t = y_true[t]

        q_t = q[:, t]   # shape = (test_set_size,)

        heavysidef = (q_t >= y_t).float()

        delta_tau = tau_grid[1] - tau_grid[0]   # Riemann sum over quantile grid step size

        crps_t = torch.sum(
            (tau_grid - heavysidef) ** 2
        ) * delta_tau

        crps_vals.append(crps_t)

    mean_crps = torch.stack(crps_vals).mean().item()

    return ecov, cal, mean_crps