import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.decomposition import PCA


class Reservoir:
    def __init__(self, input_dim, reservoir_dim, spectral_radius, seed=None):
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.w_init()

    def w_init( self ):
        if self.seed is not None:
            np.random.seed(self.seed)

        W = np.random.uniform( -1, 1, (self.reservoir_dim, self.reservoir_dim) ) 
        eigvals_res = np.linalg.eigvals( W )
        max_radius = np.max( np.abs( eigvals_res ) ) 
        self.W = W * self.spectral_radius / max_radius
        self.W_in = np.random.uniform( -1, 1, (self.reservoir_dim, self.input_dim) )

    def get_states(self, X):
        Time_steps = X.shape[0]
        states = np.zeros( (Time_steps, self.reservoir_dim) )
        s_prev = np.zeros( (1, self.reservoir_dim) )

        for t in range(Time_steps):
            x_t = X[t, :]
            # if t==0: print(self.W_in.shape, x_t.shape, s_prev.shape, self.W.shape)  # debug
            s_curr = np.tanh(self.W_in @ x_t + s_prev @ self.W )
            states[t,:] = s_curr
            s_prev = s_curr

        return states

class DataGenerator:
    def __init__(self, Time_steps, tau, n, beta, gamma, delta_t, init_steps=500 , under_samp=10):
        self.Time_steps = Time_steps*under_samp
        self.under_samp = under_samp
        self.init_steps = init_steps
        self.beta = beta
        self.gamma = gamma
        self.delta_t = delta_t
        self.tau = tau
        self.n = n

    def generate_data(self):
        delay_steps = int(self.tau / self.delta_t)
        total_steps = self.Time_steps + delay_steps + self.init_steps + 1

        x_history = np.zeros(total_steps+1)
        x_history[:delay_steps] = 1.2 + np.random.uniform(-0.1, 0.1, delay_steps )

        for t in range(delay_steps, total_steps, 1):
            x_past = x_history[t - delay_steps]
            x_curr = x_history[t]

            x_succ = x_curr + ( self.beta * x_past / ( 1 + x_past**self.n )- self.gamma * x_curr ) * self.delta_t

            x_history[t + 1] = x_succ

        skip_steps = delay_steps+self.init_steps+1

        X = x_history[skip_steps:-1][::self.under_samp]
        X = (X - np.mean(X)) / ( np.std(X) + 1e-8 )
        Y = x_history[skip_steps+1:][::self.under_samp]
        Y = (Y - np.mean(Y)) / ( np.std(Y) + 1e-8 )

        return X.reshape((-1, 1)), Y.reshape((-1, 1))
    
def get_reduced_states(Reservoir, X_train, X_test, n_components):

    states_train_high = Reservoir.get_states(X_train)
    states_test_high  = Reservoir.get_states(X_test)

    pca = PCA(n_components=n_components)
    pca.fit(states_train_high)

    states_train_low = pca.transform(states_train_high)
    states_test_low  = pca.transform(states_test_high)

    print(f"Original dim: {states_train_high.shape}")
    print(f"Reduced dim:  {states_train_low.shape}")
    
    return states_train_low, states_test_low


class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ring_levels=5):
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
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            scale = (x2 - x1) / (y2 - y1) if y2 != y1 else 1
            sdata.append((d - y1) * scale + x1)
        return sdata
        
    def plot(self, data, *args, **kwargs):
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
    
    def fill(self, data, *args, **kwargs):
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        self.ax1.legend(*args, **kwargs)


def plot_parallel_coordinates_ci(metrics_data, bounds=None, colors=None, figsize=(8, 5)):
    """ 
    Args:
        metrics_data (dict): Dictionary structured as:
            {
                'MetricName': {
                    'ModelName': [run1, run2, run3, ...],
                    ...
                },
                ...
            }
        bounds (ditc, optional): Sets the bounds on each metric's axis. If None, the bounds are tuned to the specific results.
        colors (list, optional): List of colors for each model.
        figsize (tuple, optional): Figure dimensions.
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