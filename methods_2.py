import pyro
import torch
import numpy as np
import pandas as pd

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
    """
    Calculates Empirical Coverage, Calibration Error, and mCRPS for multivariate data.
    
    Expected Shapes:
    y_true: [Time, Regions] (e.g., [330, 119])
    y_pred_samples: [Samples, Time, Regions] (e.g., [1000, 330, 119])
    """
    if not isinstance(y_true, torch.Tensor):    
        y_true = torch.from_numpy(y_true)

    # force everything to multivariate case
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
        
    if y_pred_samples.ndim == 2:
        y_pred_samples = y_pred_samples.unsqueeze(-1)

    # If the input is 3D [Samples, Time, Regions], we flatten Time and Regions 
    # to treat every prediction point across the brain equally for a global metric.
    
    num_samples = y_pred_samples.shape[0]
    # Flatten Time and Regions into a single 'observations' dimension
    # y_true_flat: [Total_Obs] 
    # y_pred_flat: [Samples, Total_Obs]
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred_samples.reshape(num_samples, -1)

    # Empirical Coverage (Global)
    Y_lower = torch.quantile(y_pred_flat, (1 - coverage_prob) / 2, dim=0)
    Y_upper = torch.quantile(y_pred_flat, (1 + coverage_prob) / 2, dim=0)

    coverage_mask = (y_true_flat >= Y_lower) & (y_true_flat <= Y_upper)
    ecov = coverage_mask.float().mean().item()

    # Calibration Error
    k_coverages = []
    for k in k_levels:
        quantile_k = torch.quantile(y_pred_flat, k, dim=0)
        k_coverage = (y_true_flat <= quantile_k).float().mean().item()
        k_coverages.append(k_coverage)

    cal = sum((tau - tau_hat)**2 for tau, tau_hat in zip(k_levels, k_coverages))

    # mCRPS (Vectorized using Pinball Loss)
    num_samples = y_pred_samples.shape[0]
    
    # quantiles grid
    tau_grid = torch.linspace(0.0, 1.0, 101)
    
    
    # shape: [101, Time, Regions]
    q = torch.quantile(y_pred_samples, tau_grid, dim=0)
    
    
    # y_true_extended has shape [1, Time, Regions] for broadcasting
    y_true_extended = y_true.unsqueeze(0)
    errors = y_true_extended - q  # [101, Time, Regions]
    
    # Pinball Loss
    # tau_grid_extended has shape [101, 1, 1] --> done for broadcasting
    tau_grid_extended = tau_grid.view(-1, 1, 1)
    loss = torch.max(tau_grid_extended * errors, (tau_grid_extended - 1) * errors)
    
    
    mcrps_per_region = 2 * loss.mean(dim=[0, 1]) # shape [Regions]
    return ecov, cal, mcrps_per_region

"""
def evaluate_metrics(y_pred_samples, y_true, coverage_prob=0.95, k_levels=[0.025, 0.5, 0.975]):
    # y_true shape = (test_set_size,)
    # y_pred_samples shape = (num_samples, test_set_size)
    if not isinstance(y_true, torch.Tensor):    
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
    q = q.mT

    #T = y_true.shape[0]
    #crps_vals = []

    
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
    
    
    y_true_aligned = y_true.view(-1, 1)    # [T, 1]
    tau_grid_aligned = tau_grid.view(1, -1) # [1, Q]
    
    errors = y_true_aligned - q  # [T, Q]

    # CRPS = 2 * Mean( PinballLoss )
    loss = torch.max(tau_grid_aligned * errors, (tau_grid_aligned - 1) * errors)
    
    crps_pointwise = 2 * torch.mean(loss, dim=1)
    mean_crps = crps_pointwise.mean().item()

    return ecov, cal, mean_crps

def quantify_tradeoff(constraints, metrics):
    results = pd.DataFrame({
        "Regime": constraints,
        "Empirical Coverage": metrics[0,:],
        "Calibration Error": metrics[1,:],
        "mCRPS": metrics[2,:]
    })
    
    i = constraints.index([0,1e1])
    results["Sharpness Degradation"] = [ ( metrics[2,i]-metrics[2,_] ).item() for _ in range(len(constraints))]
    #print(results.to_markdown(index=False))
    return results
"""