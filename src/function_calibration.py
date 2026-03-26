import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch


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
    cal_error = np.mean((predicted_cdf - quantiles[:, None]) ** 2, axis=0)

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
        iso = IsotonicRegression(out_of_bounds="clip")

        # Fit calibrator mapping empirical coverage to target nominal quantiles
        iso.fit(cdf_val[:, r], quantiles)

        new_quantiles_r = iso.transform(quantiles)
        new_quantiles_r = np.nan_to_num(new_quantiles_r, nan=0.5)
        new_quantiles_matrix[:, r] = new_quantiles_r

        # Apply Recalibration to Test Predictive Distribution
        new_q_test_r = np.array(
            [np.quantile(y_pred_test[:, :, r], q, axis=0) for q in new_quantiles_r]
        )

        cdf_test_cal[:, r] = np.mean(Y_test[:, r] <= new_q_test_r, axis=1)
        cal_error_cal[r] = np.mean((cdf_test_cal[:, r] - quantiles) ** 2)

    if plot_region is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(
            quantiles,
            cdf_test_uncal[:, plot_region],
            "-x",
            color="purple",
            label="Uncalibrated (Test)",
        )
        plt.plot(
            quantiles,
            cdf_test_cal[:, plot_region],
            "-+",
            color="red",
            label="Calibrated (Test)",
        )
        plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect Calibration")
        plt.xlabel("Target Quantile")
        plt.ylabel("Empirical Coverage")
        plt.title(f"Calibration Curve (Region {plot_region})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    return cal_error_uncal, cal_error_cal, new_quantiles_matrix


def plot_cal_test(Y_test_array, y_pred, adj_quantiles):
    region_idx = 0
    # Final Continuous Metric Evaluation on the Calibrated Test Set
    # To evaluate mCRPS/Width correctly, generate the test samples using the newly adjusted quantiles
    # Note: For scale-dependent metrics, apply  inverse z-score transformation here

    # Extract region-specific adjusted probability masses from the Isotonic matrix
    q_lower_adj = adj_quantiles[0, region_idx]
    q_upper_adj = adj_quantiles[-1, region_idx]

    # Isolate region tensors
    y_pred_test_reg = y_pred[:, :, region_idx]
    Y_test_reg = Y_test_array[:, region_idx]

    # Extract calibrated bounds utilizing the adjusted quantiles
    median_prediction_cal = torch.quantile(y_pred_test_reg, 0.5, dim=0).numpy()
    lower_bound_cal = torch.quantile(y_pred_test_reg, q_lower_adj, dim=0).numpy()
    upper_bound_cal = torch.quantile(y_pred_test_reg, q_upper_adj, dim=0).numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(median_prediction_cal, "C1--", alpha=0.9, label="Median Prediction")
    plt.plot(Y_test_reg, "k-", alpha=0.6, label="True Signal")
    plt.fill_between(
        range(len(Y_test_reg)),
        lower_bound_cal,
        upper_bound_cal,
        color="orange",
        alpha=0.3,
        label="Calibrated 95% C.I.",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Normalized signal")
    plt.title(f"Calibrated Probabilistic Forecast (Test Set Region {region_idx})")
    plt.tight_layout()
    plt.legend()
    plt.show()
