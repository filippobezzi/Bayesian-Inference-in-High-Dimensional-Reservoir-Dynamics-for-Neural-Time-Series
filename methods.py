import pyro
import torch
import numpy as np
import pandas as pd

def model(S, Y=None, beta_params=[0,10.], scale=10.):

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
        pyro.distributions.HalfNormal(torch.tensor(scale, dtype=torch.float64))
    )
    
    mu = torch.matmul(beta, S.mT)

    with pyro.plate("data", rows):
        pyro.sample(
            "obs",
            pyro.distributions.Normal(mu, sigma),
            obs=Y,
        )

def evaluate_metrics(y_pred_samples, y_true, cov_prob=0.95, k_levels=[0.025, 0.5, 0.975]):
    """
    y_pred_samples (torch.Tensor): (num_samples, length_sequence)
    y_true (torch.Tensor): (length_sequence)
    cov_prob (float, optional): C.I. to evaluate cov
    k_levels (list, optional): 

    Calculates Empirical Coverage, Calibration Error, mCRPS and Width, strictly PER-REGION (if multiDim data).
    Returns 1D PyTorch tensors of shape [Regions] for all four metrics.
    """
    if not isinstance(y_true, torch.Tensor):    
        y_true = torch.from_numpy(y_true)

    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_pred_samples.ndim == 2:
        y_pred_samples = y_pred_samples.unsqueeze(-1)

    # ==========================================
    # Per-Region Empirical Coverage
    # ==========================================
    Y_lower = torch.quantile(y_pred_samples, (1 - cov_prob) / 2, dim=0)
    Y_upper = torch.quantile(y_pred_samples, (1 + cov_prob) / 2, dim=0)

    # Mask shape: [Time, Regions]
    coverage_mask = (y_true >= Y_lower) & (y_true <= Y_upper)
    
    # Average over Time (dim=0). Result shape: [Regions]
    cov_per_region = coverage_mask.float().mean(dim=0)

    # ==========================================
    # Per-Region Calibration Error
    # ==========================================
    cal_per_region = torch.zeros(y_true.shape[1], device=y_true.device)
    
    for k in k_levels:
        quantile_k = torch.quantile(y_pred_samples, k, dim=0)
        # Average over Time (dim=0) to get coverage per region
        k_coverage = (y_true <= quantile_k).float().mean(dim=0) 
        cal_per_region += (k - k_coverage)**2

    # ==========================================
    # Per-Region mCRPS
    # ==========================================
    tau_grid = torch.linspace(0.0, 1.0, 101, dtype=y_pred_samples.dtype, device=y_pred_samples.device)
    q = torch.quantile(y_pred_samples, tau_grid, dim=0)
    
    y_true_extended = y_true.unsqueeze(0)
    errors = y_true_extended - q  
    
    tau_grid_extended = tau_grid.view(-1, 1, 1)
    loss = torch.max(tau_grid_extended * errors, (tau_grid_extended - 1) * errors)
    
    mcrps_per_region = 2 * loss.mean(dim=[0, 1])

    # ==========================================
    # Per-Region Width
    # ==========================================
    width_per_region = (Y_upper - Y_lower).mean(dim=0)

    return cov_per_region, cal_per_region, mcrps_per_region, width_per_region


def quantify_tradeoff(metrics_data, baseline_config="Rank 1"):
    metrics = list(metrics_data.keys())
    models = list(metrics_data[metrics[0]].keys())
    
    results = pd.DataFrame({
        "Configuration": models,
        "Cov": [np.mean(metrics_data["Cov"][m]) for m in models],
        "Cal": [np.mean(metrics_data["Cal"][m]) for m in models],
        "mCRPS": [np.mean(metrics_data["mCRPS"][m]) for m in models],
        "Width": [np.mean(metrics_data["Width"][m]) for m in models]
    })
    
    baseline_idx = models.index(baseline_config)
    
    results["mCRPS (Total Degradation)"] = results["mCRPS"] - results["mCRPS"].iloc[baseline_idx]
    results["Width (Sharpness Degradation)"] = results["Width"] - results["Width"].iloc[baseline_idx]

    return results