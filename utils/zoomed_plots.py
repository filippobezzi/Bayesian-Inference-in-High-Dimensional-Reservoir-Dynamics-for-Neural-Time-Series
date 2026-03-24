import numpy as np
import matplotlib.pyplot as plt

def zoomed_plots(region_idx, y_mean, y_true, upper, lower, steps=300, n_plots=4, rows=2):
    
    cols = int(np.ceil(n_plots / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(13, 8)) 
    fig.suptitle(f"Region {region_idx}", fontsize = 16)
    
    axes = ax.flatten()

    distances = np.arange(steps, len(y_mean[:,0]) + steps - 1, steps, dtype=int)   
    
    for i, distance in zip(range(n_plots), distances): 
        
        start_idx = distance - steps
        
        # Extract the sliced data
        y_true_slice = y_true[start_idx:distance, region_idx]
        y_mean_slice = y_mean[start_idx:distance, region_idx]
        lower_slice = lower[start_idx:distance, region_idx]
        upper_slice = upper[start_idx:distance, region_idx]
        
        # Calculate the actual end index (protects against the final slice being shorter)
        actual_end_idx = start_idx + len(y_true_slice)
        
        # Set x_vals to the actual absolute time steps
        x_vals = np.arange(start_idx, actual_end_idx) 
        
        # Plotting
        axes[i].plot(x_vals, y_true_slice, label="True Signal", color="black", alpha=0.6)
        axes[i].plot(x_vals, y_mean_slice, label="Bayesian Median", color="blue")

        axes[i].fill_between(
            x_vals, 
            lower_slice, 
            upper_slice, 
            color="blue", alpha=0.2, label="95% Confidence Interval"
        )

        #title showing the range
        axes[i].set_title(f"Steps: {start_idx} to {actual_end_idx}")
        
        axes[i].set_xlabel("Time Steps")
        axes[i].set_ylabel("Amplitude")
        
    # Hide any extra empty subplots if n_plots isn't a perfect multiple of rows*cols
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()