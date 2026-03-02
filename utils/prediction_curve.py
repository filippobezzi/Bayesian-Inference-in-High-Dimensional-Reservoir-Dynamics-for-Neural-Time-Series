from utils.ESNDataset import ESNDataset
from utils.ESNVariational import ESNVariational
import torch
import matplotlib.pyplot as plt


def prediction_curve(esnvariational_object: ESNVariational, 
                     test_dataset: ESNDataset,
                     num_samples: int, 
                     scaler: dict = None,
                     plot: bool = True, 
                     region_idx: int = 0, 
                     title: str = "Predicted vs Actual Data"):
    
    # y_samples shape: [Num_Samples, Time, Num_Regions]
    y_samples = esnvariational_object.predict(test_states=test_dataset.states, num_samples=num_samples)
    y_true = test_dataset.predictions 
    
    # 1. IN-PLACE RESCALING: Prevents sudden RAM doubling
    if scaler:
        m = scaler['mean']
        s = scaler['std']
        
        # Modify the giant tensor by overwriting its own memory
        y_samples.mul_(s).add_(m)
        
        if isinstance(y_true, torch.Tensor):
            y_true.mul_(s).add_(m)
        else:
            y_true = (y_true * s) + m

    # 2. VECTORIZED QUANTILES: Executed in a single pass
    # Define the 3 levels we need: [2.5%, 50% (median), 97.5%]
    q_levels = torch.tensor([0.025, 0.5, 0.975], dtype=y_samples.dtype, device=y_samples.device)
    
    # q_vals will have shape [3, Time, Num_Regions]
    q_vals = torch.quantile(y_samples, q_levels, dim=0)
    
    lower_bound = q_vals[0]
    mean_pred = q_vals[1]  # 50th percentile is the median
    upper_bound = q_vals[2]

    # Plotting logic
    if plot:
        if mean_pred.ndim > 1:
            y_plot_true = y_true[:, region_idx]
            y_plot_mean = mean_pred[:, region_idx]
            y_plot_lower = lower_bound[:, region_idx]
            y_plot_upper = upper_bound[:, region_idx]
            label_suffix = f" (Region {region_idx})"
        else:
            y_plot_true = y_true
            y_plot_mean = mean_pred
            y_plot_lower = lower_bound
            y_plot_upper = upper_bound
            label_suffix = ""

        # Use 'fig' so we can explicitly close it to avoid Matplotlib memory leaks
        fig = plt.figure(figsize=(12, 5))
        
        plt.plot(y_plot_true, label=f"True Signal {label_suffix}", color="black", alpha=0.6)
        plt.plot(y_plot_mean, label="Bayesian Median", color="blue")

        plt.fill_between(
            range(len(mean_pred)), 
            y_plot_lower, 
            y_plot_upper, 
            color="blue", alpha=0.2, label="95% Confidence Interval"
        )

        plt.title(f"{title} {label_suffix}")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        
        # Forcefully close the figure to free Matplotlib's RAM
        plt.close(fig)

    return y_true, y_samples, mean_pred, lower_bound, upper_bound