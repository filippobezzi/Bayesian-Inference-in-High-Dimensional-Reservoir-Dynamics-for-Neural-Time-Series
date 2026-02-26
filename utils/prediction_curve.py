from utils.ESNDataset import ESNDataset
from utils.ESNVariational import ESNVariational
import torch
import matplotlib.pyplot as plt


import torch
import matplotlib.pyplot as plt
from utils.ESNDataset import ESNDataset
from utils.ESNVariational import ESNVariational

def prediction_curve(esnvariational_object: ESNVariational, 
                     test_dataset: ESNDataset, 
                     scaler: dict = None,
                     plot: bool = True, 
                     region_idx: int = 0, 
                     title: str = "Predicted vs Actual Data"):
    
    # y_samples shape: [Num_Samples, Time, Num_Regions] (e.g., [1000, 259, 119])
    y_samples = esnvariational_object.predict(test_states=test_dataset.states)
    
    # y_true shape: [Time, Num_Regions] (e.g., [259, 119])
    y_true = test_dataset.predictions 
    
    # Rescale data back to original physical values
    if scaler:
        m = scaler['mean']
        s = scaler['std']
        # Broadcasting handles the multiplication/addition
        y_samples = (y_samples * s) + m
        y_true = (y_true * s) + m

    # Calculate summary statistics along the sample dimension (dim=0)
    # The resulting shapes will be [Time, Num_Regions]
    mean_pred = y_samples.median(dim=0).values
    lower_bound = torch.quantile(y_samples, 0.025, dim=0) # 95% Confidence Interval (Lower)
    upper_bound = torch.quantile(y_samples, 0.975, dim=0) # 95% Confidence Interval (Upper)

    if mean_pred.ndim > 1:
        y_plot_true = y_true[:, region_idx]
        y_plot_mean = mean_pred[:, region_idx]
        y_plot_lower = lower_bound[:, region_idx]
        y_plot_upper = upper_bound[:, region_idx]
        label_suffix = f" (Region {region_idx})"
    else:
        # Data is already 1D
        y_plot_true = y_true
        y_plot_mean = mean_pred
        y_plot_lower = lower_bound
        y_plot_upper = upper_bound
        label_suffix = ""

    # Plot the results for the specifically requested region
    if plot:
        plt.figure(figsize=(12, 5))
        
        # Slice the tensors to extract only the target region [:, region_idx]
        plt.plot(y_plot_true, label=f"True Signal {label_suffix}", color="black", alpha=0.6)
        plt.plot(y_plot_mean, label="Bayesian Median", color="blue")

        # Plot the uncertainty bands
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

    # Return the full multi-dimensional tensors for metric evaluation
    return y_true, y_samples, mean_pred, lower_bound, upper_bound