import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator

def plot_probabilistic_forecast(y_true, y_pred, y_lower, y_upper, plot_title="Probabilistic forecast", num_steps=None):
    
    # Apply a zoom if 'num_steps' is specified
    if num_steps is not None:
        y_true = y_true[:num_steps]
        y_pred = y_pred[:num_steps]
        y_lower = y_lower[:num_steps]
        y_upper = y_upper[:num_steps]
        
    time_steps = np.arange(len(y_true))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(time_steps, y_true, color="black", linewidth=1.2, label="True value")
    ax.plot(time_steps, y_pred, color="blue", linewidth=1.2, label="Prediction")
    ax.fill_between(time_steps, y_lower, y_upper, color="blue", alpha=0.2, label=r"95% CI")
    
    ax.set_title(plot_title)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Signal amplitude")
    ax.legend(loc="best", frameon=False)
    ax.tick_params(direction="in")

    plt.tight_layout()
    plt.show()


def plot_functional_connectivity(y_true_multi, y_pred_multi, title="Functional Connectivity Comparison"):

    # Compute correlation matrices
    corr_true = np.corrcoef(np.asarray(y_true_multi).T)
    corr_pred = np.corrcoef(np.asarray(y_pred_multi).T)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1, 0.05]})
    fig.suptitle(title)
    
    # Heatmap real data
    sns.heatmap(corr_true, vmin=-1, vmax=1, cmap="vlag", ax=ax[0], cbar=False)
    ax[0].set_title("True")
    ax[0].set_xlabel("Regions")
    ax[0].set_ylabel("Regions")
    
    # Heatmap prediction
    sns.heatmap(corr_pred, vmin=-1, vmax=1, cmap="vlag", ax=ax[1], cbar=True, cbar_ax=ax[2])
    ax[1].set_title("Predicted")
    ax[1].set_xlabel("Regions")
    
    plt.tight_layout()
    plt.show()

#########################################################################
#                           METRICS PLOTS                               # 
#########################################################################

class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ring_levels=5):
        """
        Initializes a multi-axis radar chart capable of handling independent scales for each variable.

        Args:
            fig (matplotlib.figure.Figure): The parent matplotlib figure object.
            variables (list of str): The names of the metrics to be plotted on each axis.
            ranges (list of tuple): The minimum and maximum limits for each corresponding variable.
            n_ring_levels (int, optional): The number of concentric grid lines. Defaults to 5.
        """
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True) for _ in range(len(variables)+1)]
        
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)
        
        for i, ax in enumerate(axes):
            j = 0 if (i==0 or i==1) else i-1
            ax.set_ylim(*ranges[j])
            grid = np.linspace(*ranges[j], num=n_ring_levels, endpoint=False)
            gridlabel = [f"{round(x,2)}" for x in grid]
            gridlabel[0] = "" 
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[j])
            
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        
        self.ax.yaxis.grid()
        self.ax.xaxis.grid()
        self.ax.spines['polar'].set_visible(True)
        
        self.ax1.axis('off')
        self.ax1.set_zorder(9)
        
        self.ax.set_thetagrids(angles, labels=variables)
        self.ax.tick_params(axis='both', pad=15)

    def _scale_data(self, data, ranges):
        """
        Normalizes input data across distinct metric ranges to align with the primary radial axis.

        Args:
            data (list of float): The raw data points matching the variables order.
            ranges (list of tuple): The min-max bounds for scaling calculations.

        Returns:
            list of float: The standardized data points mapped to the baseline axis scale.
        """
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            scale = (x2 - x1) / (y2 - y1) if y2 != y1 else 1
            sdata.append((d - y1) * scale + x1)
        return sdata
        
    def plot(self, data, *args, **kwargs):
        """
        Plots a 1D array of values onto the scaled radar axes as a connecting line.

        Args:
            data (list of float): The data points to map onto the radar chart.
            *args: Standard matplotlib plot positional arguments.
            **kwargs: Standard matplotlib plot keyword arguments.
        """
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
    
    def fill(self, data, *args, **kwargs):
        """
        Fills the polygonal area enclosed by the plotted data points on the radar chart.

        Args:
            data (list of float): The data points defining the polygon vertices.
            *args: Standard matplotlib fill positional arguments.
            **kwargs: Standard matplotlib fill keyword arguments.
        """
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """
        Binds a matplotlib legend to the internal radar axis.

        Args:
            *args: Standard matplotlib legend positional arguments.
            **kwargs: Standard matplotlib legend keyword arguments.
        """
        self.ax1.legend(*args, **kwargs)


def plot_parallel_coordinates_ci(metrics_data, bounds=None, colors=None, figsize=(8, 5)):
    """ 
    Constructs a parallel coordinates plot with interval shading to visualize multi-metric performance distributions across different models.

    Args:
        metrics_data (dict): Dictionary structured as {'MetricName': {'ModelName': [run1, run2, ...], ...}, ...}.
        bounds (dict, optional): Sets the min and max bounds on each metric's vertical axis. If None, the bounds are dynamically tuned. Defaults to None.
        colors (list, optional): List of string or hex colors for each model line and interval fill. Defaults to None.
        figsize (tuple, optional): Dimensions of the generated figure. Defaults to (8, 5).
    """

    metrics = list(metrics_data.keys())
    models = list(metrics_data[metrics[0]].keys())
    n_metrics = len(metrics)
    
    if colors is None:
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    # Compute bounds and scaling limits
    data_mean = {m: np.array([np.mean(metrics_data[met][m]) for met in metrics]) for m in models}
    data_upper = {m: np.array([np.max(metrics_data[met][m]) for met in metrics]) for m in models}
    data_lower = {m: np.array([np.min(metrics_data[met][m]) for met in metrics]) for m in models}

    if bounds is not None:
        ymins = [bounds[met][0] for met in metrics]
        ymaxs = [bounds[met][1] for met in metrics]
    else:
        ymins = [min([data_lower[m][i] for m in models]) for i in range(n_metrics)]
        ymaxs = [max([data_upper[m][i] for m in models]) for i in range(n_metrics)]

    # Add padding to axes
    ymins = [y - 0.05 * (ymaxs[i] - y) if ymaxs[i] != y else y - 0.05 for i, y in enumerate(ymins)]
    ymaxs = [y + 0.05 * (y - ymins[i]) if ymins[i] != y else y + 0.05 for i, y in enumerate(ymaxs)]

    def scale(val, i):
        return (val - ymins[i]) / (ymaxs[i] - ymins[i])

    fig, host = plt.subplots(figsize=figsize)
    axes = [host] + [host.twinx() for _ in range(n_metrics - 1)]

    # Axis superposition and scaling
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])

        if metrics[i] == 'Width':
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (n_metrics - 1)))

    host.set_xlim(0, n_metrics - 1)
    host.set_xticks(range(n_metrics))
    host.set_xticklabels(metrics, fontsize=12)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    lin = np.linspace(0, n_metrics - 1, n_metrics)

    # Path rendering
    for j, model in enumerate(models):
        c = colors[j % len(colors)]
        
        scaled_lower = np.array([scale(data_lower[model][i], i) for i in range(n_metrics)])
        scaled_upper = np.array([scale(data_upper[model][i], i) for i in range(n_metrics)])
        scaled_mean = np.array([scale(data_mean[model][i], i) for i in range(n_metrics)])

        # Construct closed polygon for interval shading
        verts_fill = np.concatenate([
            np.stack([lin, scaled_upper]).T,
            np.flip(np.stack([lin, scaled_lower]).T, axis=0),
            [[lin[0], scaled_upper[0]]]
        ])
        codes_fill = [Path.MOVETO] + [Path.LINETO] * (len(verts_fill) - 1)
        
        patch_fill = patches.PathPatch(Path(verts_fill, codes_fill), facecolor=c, alpha=0.2, lw=0)
        host.add_patch(patch_fill)

        # Construct mean line
        verts_mean = np.stack([lin, scaled_mean]).T
        host.plot(verts_mean[:, 0], verts_mean[:, 1], color=c, lw=2, label=model)

    host.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(models))
    plt.tight_layout()
    plt.show()