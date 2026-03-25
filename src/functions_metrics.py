import pyro
import torch
import numpy as np
import pandas as pd


def evaluate_metrics(y_pred_samples, y_true, coverage_prob=0.95, k_levels=np.linspace(0.0,1.0,int(1/0.025))):
    """
    Calculates Empirical Coverage, Calibration Error, and mCRPS strictly PER-REGION.
    Returns 1D PyTorch tensors of shape [Regions] for all three metrics.
    """
    if not isinstance(y_true, torch.Tensor):    
        y_true = torch.from_numpy(y_true)

    # Force everything to multivariate (3D) case
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_pred_samples.ndim == 2:
        y_pred_samples = y_pred_samples.unsqueeze(-1)

    # We DO NOT flatten the tensors here. We keep the [Time, Regions] structure.

    # ==========================================
    # Per-Region Empirical Coverage
    # ==========================================
    Y_lower = torch.quantile(y_pred_samples, (1 - coverage_prob) / 2, dim=0)
    Y_upper = torch.quantile(y_pred_samples, (1 + coverage_prob) / 2, dim=0)

    # Mask shape: [Time, Regions]
    coverage_mask = (y_true >= Y_lower) & (y_true <= Y_upper)
    
    # Average over Time (dim=0). Result shape: [Regions]
    ecov_per_region = coverage_mask.float().mean(dim=0)

    # ==========================================
    # Per-Region Calibration Error
    # ==========================================
    cal_per_region = torch.zeros(y_true.shape[1], device=y_true.device)
    
    for k in k_levels:
        quantile_k = torch.quantile(y_pred_samples, k, dim=0)
        # Average over Time (dim=0) to get coverage per region
        k_coverage = (y_true <= quantile_k).float().mean(dim=0) 
        cal_per_region += (k - k_coverage)**2

    cal_per_region /= len(k_levels)
    # ==========================================
    # Per-Region mCRPS (Pinball Loss)
    # ==========================================
    tau_grid = torch.linspace(0.0, 1.0, 101, device=y_pred_samples.device)
    q = torch.quantile(y_pred_samples, tau_grid, dim=0)
    
    y_true_extended = y_true.unsqueeze(0)
    errors = y_true_extended - q  
    
    tau_grid_extended = tau_grid.view(-1, 1, 1)
    loss = torch.max(tau_grid_extended * errors, (tau_grid_extended - 1) * errors)
    
    mcrps_per_region = 2 * loss.mean(dim=[0, 1])

    # ==========================================
    # Per-Region Mean Interval Width (MPIW)
    # ==========================================
    
    # Calculate the width at each time step for each region
    interval_widths = Y_upper - Y_lower  # Shape: [Time, Regions]
    
    # Average over Time (dim=0). Result shape: [Regions]
    width_per_region = interval_widths.mean(dim=0)
    
    return ecov_per_region, cal_per_region, mcrps_per_region, width_per_region

import torch

def evaluate_metrics_qr(y_pred_quantiles, y_true, tau_levels=[0.025, 0.25, 0.5, 0.75, 0.975]):
    """
    Calculates metrics for Quantile Regression models.
    Expects y_pred_quantiles shape: [Time, Quantiles, Regions] -> e.g., [230, 5, 119]
    Expects y_true shape: [Time, Regions] -> e.g., [230, 119]
    """
    if not isinstance(y_true, torch.Tensor):    
        y_true = torch.from_numpy(y_true)

    # Force y_true to 2D [Time, Regions] if it isn't already
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)

    # ==========================================
    # Per-Region Empirical Coverage & Width
    # ==========================================
    # Assuming tau_levels are sorted ascending. 
    # Index 0 is the lower bound, Index -1 is the upper bound.
    Y_lower = y_pred_quantiles[:, 0, :]   # Shape: [Time, Regions]
    Y_upper = y_pred_quantiles[:, -1, :]  # Shape: [Time, Regions]

    coverage_mask = (y_true >= Y_lower) & (y_true <= Y_upper)
    ecov_per_region = coverage_mask.float().mean(dim=0)
    width_per_region = (Y_upper - Y_lower).mean(dim=0)

    # ==========================================
    # Per-Region Calibration Error
    # ==========================================
    # We can only evaluate calibration at the 5 specific quantiles predicted
    cal_per_region = torch.zeros(y_true.shape[1], device=y_true.device)
    
    for i, tau in enumerate(tau_levels):
        q_pred = y_pred_quantiles[:, i, :] # The specific quantile predictions
        tau_coverage = (y_true <= q_pred).float().mean(dim=0) 
        cal_per_region += (tau - tau_coverage)**2

    cal_per_region /= len(tau_levels)

    # ==========================================
    # Per-Region mCRPS (Pinball Loss)
    # ==========================================
    # Reshape y_true to [Time, 1, Regions] to broadcast with our 5 quantiles
    y_true_extended = y_true.unsqueeze(1)
    errors = y_true_extended - y_pred_quantiles  # Shape: [Time, Quantiles, Regions]
    
    # Reshape tau_levels to [1, Quantiles, 1] for broadcasting
    tau_tensor = torch.tensor(tau_levels, device=y_pred_quantiles.device).view(1, -1, 1)
    
    # Calculate Pinball loss across the 5 quantiles
    loss = torch.max(tau_tensor * errors, (tau_tensor - 1) * errors)
    
    # Average over Time (dim=0) and Quantiles (dim=1)
    mcrps_per_region = 2 * loss.mean(dim=[0, 1])
    
    return ecov_per_region, cal_per_region, mcrps_per_region, width_per_region

def print_metrics(y_true, ecov_per_region, cal_per_region, mcrps_per_region, width_per_region):
    """
    Prints the metrics results as scalars, i.e. averaged over all regions
    """
    # ranges for each region (used to normalize the mcrps)
    ranges = y_true.max(dim=0).values - y_true.min(dim=0).values


    # convert the vector of mrcps in a normalized average
    n_mcrps = mcrps_per_region / ranges

    avg_mcrps = n_mcrps.mean().item()

    std_ecov = np.std(ecov_per_region.detach().numpy())/np.sqrt(len(ecov_per_region))
    std_cal = np.std(cal_per_region.detach().numpy())/np.sqrt(len(cal_per_region))
    std_mcrps = np.std(n_mcrps.detach().numpy())/np.sqrt(len(n_mcrps))
    std_width = np.std(width_per_region.detach().numpy())/np.sqrt(len(width_per_region))

    # Rounding
    print(f"Empirical Coverage: {ecov_per_region.mean().item():.3f} ± {std_ecov:.3f}")
    print(f"Calibration Error:  {cal_per_region.mean().item():.5f} ± {std_cal:.5f}")
    print(f"mCRPS (Normalized): {avg_mcrps:.4f} ± {std_mcrps:.4f}")
    print(f"Width: {width_per_region.mean().item():.3f} ± {std_width:.3f}")


    return


def evaluate_metrics_multiple(esn_obj, y_true, test_set, n_trials=5, model_type = None, tau_levels = None):
    """
    Runs multiple trials (predictions) for a SINGLE model object.
    Returns a dictionary of lists containing the metrics across all trials.
    """
    ranges = y_true.max(dim=0).values - y_true.min(dim=0).values
    
    # Initialize the dictionary for this specific method
    method_results = {"Cal": [], "Cov": [], "mCRPS": [], "Width": []}

    for i in range(n_trials):
        # 1. Generate Prediction for this trial
        # (Simplified to one line since the methods behave identically)
        y_pred = esn_obj.predict(test_set).cpu()

        if model_type != 'QR':
        # Calculate Metrics 
            ecov, cal, mcrps, width = evaluate_metrics(y_pred, y_true)
        else: 
            ecov, cal, mcrps, width = evaluate_metrics_qr(y_pred, y_true, tau_levels = tau_levels)
        # Normalize mCRPS
        n_mcrps = mcrps / ranges

        # Append the mean values to our method's dictionary
        method_results["Cov"].append(ecov.mean().item())
        method_results["Cal"].append(cal.mean().item())
        method_results["mCRPS"].append(n_mcrps.mean().item())
        method_results["Width"].append(width.mean().item())
        
    return method_results

def combine_metrics_dictionaries(calculated_results_dict):
    """
    Takes a dictionary of pre-calculated method results and flips the nesting 
    to match the final required JSON-like structure.
    """
    final_output = {
        "Cal": {},
        "Cov": {},
        "mCRPS": {},
        "Width": {}
    }
    
    for method_name, method_data in calculated_results_dict.items():
        
        # method_data is your dict: {"Cal": [...], "Cov": [...], ...}
        # We map each list to its corresponding place in the final output
        final_output["Cal"][method_name] = method_data["Cal"]
        final_output["Cov"][method_name] = method_data["Cov"]
        final_output["mCRPS"][method_name] = method_data["mCRPS"]
        final_output["Width"][method_name] = method_data["Width"]
        
    return final_output