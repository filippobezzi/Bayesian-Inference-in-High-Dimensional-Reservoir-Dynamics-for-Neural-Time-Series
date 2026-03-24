import numpy as np
import torch
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

################### DATA ###################
def partition_states(S, Y, block_size, buffer_size=0):
    """
    Splits continuous reservoir states and target sequences into discrete, temporally ordered blocks separated by an optionial buffer to mitigate autocorrelation leakage.

    Args:
        S (ndarray): Reservoir state matrix of shape [Time, Reservoir_Dim].
        Y (ndarray): Target variable matrix of shape [Time, Target_Dim].
        block_size (int): Temporal length of each extracted continuous block.
        buffer_size (int, optional): Number of time steps to discard between consecutive blocks to enforce temporal independence. Defaults to 0.

    Returns:
        tuple: (S_blocks, Y_blocks) containing lists of partitioned state arrays and target arrays.
     """
    S_blocks, Y_blocks = [], []
    start_idx = 0
    while start_idx < S.shape[0]: # Add blocks until there is data
        if start_idx + block_size <= S.shape[0]: end_idx = start_idx + block_size
        else: end_idx = S.shape[0]
        S_blocks.append(S[start_idx:end_idx])
        Y_blocks.append(Y[start_idx:end_idx])
        start_idx = end_idx + buffer_size
    return S_blocks, Y_blocks

def z_rescale_tensor(X_tensor, mean, std):
    """
    Reverses the z-score standardization on a PyTorch tensor to restore the original scale of the data using pre-computed empirical moments.

    Args:
        X_tensor (torch.Tensor): Standardized tensor to be rescaled.
        mean (ndarray | float): Empirical mean used during the original standardization.
        std (ndarray | float): Empirical standard deviation used during the original standardization.

    Returns:
        torch.Tensor: Scale-restored tensor with dimensions matching X_tensor.
    """
    mean_t = torch.tensor(mean, dtype=X_tensor.dtype)
    std_t = torch.tensor(std, dtype=X_tensor.dtype)
    return (X_tensor * std_t) + mean_t

################### RC ###################
def get_reduced_states(Reservoir, X, n_components):
    """
    Applies principal component analysis (PCA) to compress the dimensionality of the reservoir's internal RNN states.

    Args:
        Reservoir (object): Instantiated reservoir computer object containing the get_states method.
        X (ndarray): Input time-series array used to drive the reservoir dynamics. Shape [Time, Input_Dim].
        n_components (int): Target number of orthogonal principal components to retain.

    Returns:
        ndarray: Dimensionally reduced state matrix of shape [Time, n_components].
    """
    states_high = Reservoir.get_states(X)

    pca = PCA(n_components=n_components)
    pca.fit(states_high)

    states_low = pca.transform(states_high)

    print(f"Original dim: {states_high.shape}")
    print(f"Reduced dim:  {states_low.shape}")
    
    return states_low

################### METRICS ###################

def quantify_tradeoff(metrics_data, baseline_config="Rank 1"):
    """
    Aggregates grid search metric evaluations and calculates the relative performance degradation against a designated baseline configuration.

    Args:
        metrics_data (dict): Nested dictionary mapping metric names ('Cov', 'Cal', 'mCRPS', 'Width') to model configuration ranks to arrays of regional scores.
        baseline_config (str, optional): The dictionary key representing the baseline model for delta calculations. Defaults to "Rank 1".

    Returns:
        pd.DataFrame: Tabular summary of mean metrics per configuration, including delta values for mCRPS and Width.
    """
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

def evaluate_metrics(y_pred_samples, y_true, cov_prob=0.95, quantiles=None):
    """
    Computes empirical coverage, mean squared calibration error (MSCE), mean Continuous Ranked Probability Score (mCRPS), and prediction interval width across spatial regions.

    Args:
        y_pred_samples (torch.Tensor): Posterior predictive samples of shape [Samples, Time, Regions].
        y_true (torch.Tensor | ndarray): Ground truth observations of shape [Time, Regions].
        cov_prob (float, optional): Target probability mass for the central prediction interval. Defaults to 0.95.
        quantiles (torch.Tensor | ndarray, optional): Target quantile thresholds for calibration evaluation. Accepts a 1D global array or 2D region-specific matrix. Defaults to 19 evenly spaced quantiles.

    Returns:
        tuple: (cov_per_region, cal_per_region, mcrps_per_region, width_per_region) containing PyTorch tensors of spatial metrics.
    """
    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, 19)

    if not isinstance(y_true, torch.Tensor):    
        y_true = torch.from_numpy(y_true)

    # Adjust PyTorch tensor dims for 1 dimensional datasets
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_pred_samples.ndim == 2:
        y_pred_samples = y_pred_samples.unsqueeze(-1)

    # Convert quantiles to a PyTorch tensor
    if not isinstance(quantiles, torch.Tensor):
        quantiles_t = torch.tensor(quantiles, dtype=y_pred_samples.dtype, device=y_pred_samples.device)
    else:
        quantiles_t = quantiles

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
    num_regions = y_true.shape[1]
    cal_per_region = torch.zeros(num_regions, device=y_true.device)
    
    # Handle both 1D (Global Quantiles) and 2D (Regional Quantiles)
    if quantiles_t.ndim == 1:
        for q in quantiles_t:
            # Get float value via .item()
            quantile_vals = torch.quantile(y_pred_samples, q.item(), dim=0)   # Compute quantile across samples for fixed TimeStep
            q_coverage = (y_true <= quantile_vals).float().mean(dim=0)     
            cal_per_region += (q - q_coverage)**2   # Squared Calibration Error
        
        # Mean Squared Calibration Error
        cal_per_region /= len(quantiles_t)
        
    elif quantiles_t.ndim == 2:
        num_quantiles = quantiles_t.shape[0]
        # Evaluate Region-by-Region to handle independent quantile sequences
        for r in range(num_regions):
            for i in range(num_quantiles):
                q = quantiles_t[i, r]
                # Extract all samples and all time steps for region r
                quantile_val = torch.quantile(y_pred_samples[:, :, r], q.item(), dim=0) # Compute quantile across samples for fixed TimeStep, r
                q_coverage = (y_true[:, r] <= quantile_val).float().mean(dim=0)
                cal_per_region[r] += (q - q_coverage)**2    # Squared Calibration Error
                
        # Mean Squared Calibration Error
        cal_per_region /= num_quantiles

    # ==========================================
    # Per-Region mCRPS
    # ==========================================
    tau_grid = torch.linspace(0.0, 1.0, 101, dtype=y_pred_samples.dtype, device=y_pred_samples.device)
    q_mcrps = torch.quantile(y_pred_samples, tau_grid, dim=0)   # Shape: [Quantiles, TimeSteps, Regions]
    
    y_true_extended = y_true.unsqueeze(0)   # Shape: [1, TimeSteps, Regions]
    errors = y_true_extended - q_mcrps  
    
    tau_grid_extended = tau_grid.view(-1, 1, 1) # Shape: [Quantiles, 1, 1]
    loss = torch.max(tau_grid_extended * errors, (tau_grid_extended - 1) * errors) # Shape: [Quantiles, TimeSteps, Regions]
    
    mcrps_per_region = 2 * loss.mean(dim=[0, 1])    # Shape: [Regions]

    # ==========================================
    # Per-Region Width
    # ==========================================
    width_per_region = (Y_upper - Y_lower).mean(dim=0) # Shape: [Regions,] 

    return cov_per_region, cal_per_region, mcrps_per_region, width_per_region

def check_calibration(q, Y, quantiles):
    """
    Evaluates the empirical Cumulative Distribution Function (CDF) and computes the mean squared calibration error (MSCE) against target nominal quantiles.

    Args:
        q (ndarray): Predicted quantile matrix of shape [Quantiles, Time, Regions].
        Y (ndarray): Ground truth observations of shape [Time, Regions].
        quantiles (ndarray): Target nominal quantiles of shape [Quantiles].

    Returns:
        tuple: (cal_error, predicted_cdf) containing the per-region MSCE array and the empirical coverage matrix.
    """
    # Y is broadcasted against q. Resulting mask shape: [Quantiles, TimeSteps, Regions]
    coverage_mask = Y <= q
    
    # Average over TimeSteps to isolate per-region empirical CDF. Shape: [Quantiles, Regions]
    predicted_cdf = np.mean(coverage_mask, axis=1)
    
    # Compute error per region. Shape: [Regions]
    cal_error = np.mean((predicted_cdf - quantiles[:, None])**2, axis=0)
    
    return cal_error, predicted_cdf

def calibrate(y_pred_test, y_pred_val, Y_test, Y_val, quantiles=None, plot_region=0):
    """
    Executes Isotonic Regression on validation predictions to map empirical coverage to nominal quantiles, outputting recalibrated quantile boundaries for the test set.

    Args:
        y_pred_test (ndarray): Uncalibrated predictive samples for the test sequence of shape [Samples, Time, Regions].
        y_pred_val (ndarray): Uncalibrated predictive samples for the validation sequence of shape [Samples, Time, Regions].
        Y_test (ndarray): Ground truth test observations of shape [Time, Regions].
        Y_val (ndarray): Ground truth validation observations of shape [Time, Regions].
        quantiles (ndarray, optional): Target nominal quantiles for calibration mapping. Defaults to 99 percentiles.
        plot_region (int | None, optional): Spatial region index to visualize the calibration curve. Set to None to disable plotting. Defaults to 0.

    Returns:
        tuple: (cal_error_uncal, cal_error_cal, new_quantiles_matrix) containing original MSCE, recalibrated MSCE, and the region-adjusted nominal quantile matrix.
    """
    if quantiles is None:
        quantiles = np.linspace(0.01, 0.99, 99)
        
    regions = Y_test.shape[1]
    
    # Uncalibrated Test Evaluation
    q_test_uncal = np.quantile(y_pred_test, quantiles, axis=0)
    cal_error_uncal, cdf_test_uncal = check_calibration(q_test_uncal, Y_test, quantiles)
    
    # Extract Validation Empirical CDF for Fitting
    q_val = np.quantile(y_pred_val, quantiles, axis=0)
    _, cdf_val = check_calibration(q_val, Y_val, quantiles)
    
    new_quantiles_matrix = np.zeros((len(quantiles), regions))
    cal_error_cal = np.zeros(regions)
    cdf_test_cal = np.zeros((len(quantiles), regions))
    
    for r in range(regions):
        iso = IsotonicRegression(out_of_bounds='clip')
        
        # Fit calibrator mapping empirical coverage to target nominal quantiles
        iso.fit(cdf_val[:, r], quantiles)
        
        new_quantiles_r = iso.transform(quantiles)
        new_quantiles_r = np.nan_to_num(new_quantiles_r, nan=0.5)
        new_quantiles_matrix[:, r] = new_quantiles_r
        
        # Apply Recalibration to Test Predictive Distribution
        new_q_test_r = np.array([np.quantile(y_pred_test[:, :, r], q, axis=0) for q in new_quantiles_r])
        
        cdf_test_cal[:, r] = np.mean(Y_test[:, r] <= new_q_test_r, axis=1)
        cal_error_cal[r] = np.mean((cdf_test_cal[:, r] - quantiles)**2)

    if plot_region is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(quantiles, cdf_test_uncal[:, plot_region], '-x', color='purple', label='Uncalibrated (Test)')
        plt.plot(quantiles, cdf_test_cal[:, plot_region], '-+', color='red', label='Calibrated (Test)')
        plt.plot([0,1],[0,1],'--', color='grey', label='Perfect Calibration')
        plt.xlabel('Target Quantile')
        plt.ylabel('Empirical Coverage')
        plt.title(f'Calibration Curve (Region {plot_region})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return cal_error_uncal, cal_error_cal, new_quantiles_matrix