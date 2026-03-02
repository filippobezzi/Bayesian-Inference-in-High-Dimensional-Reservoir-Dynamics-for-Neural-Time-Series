import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyro
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer import Predictive
import pyro.distributions as dist
from pyro.optim import Adam
import random
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.graph_objects as go

y_true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
y_mean_np = y_mean.detach().cpu().numpy() if torch.is_tensor(y_mean) else np.array(y_mean)
upper_np = upper.detach().cpu().numpy() if torch.is_tensor(upper) else np.array(upper)
lower_np = lower.detach().cpu().numpy() if torch.is_tensor(lower) else np.array(lower)

# Extract dimensions
time_len = y_true_np.shape[0]
n_regions = y_true_np.shape[1]
time_steps = np.arange(time_len)

fig = go.Figure()

# --- PREPARE CONFIDENCE INTERVAL POLYGON (Region 0) ---
x_ci = np.concatenate([time_steps, time_steps[::-1]])
# Slice [:, 0] gets all time steps for Region 0. [::-1] safely reverses the NumPy array.
y_ci_initial = np.concatenate([upper_np[:, 0], lower_np[:, 0][::-1]])

# Trace 0: Confidence Interval
fig.add_trace(go.Scatter( 
    x=x_ci,
    y=y_ci_initial,
    fill='toself',
    fillcolor='rgba(65, 105, 225, 0.5)', # Highly visible Royal Blue
    line=dict(color='rgba(65, 105, 225, 0.8)', width=1), # Distinct border
    name='95% Confidence Interval'
))

# Trace 1: True Signal
fig.add_trace(go.Scattergl(
    x=time_steps, y=y_true_np[:, 0],
    mode='lines',
    line=dict(color='rgba(40, 40, 40, 0.9)', width=1.5), # Dark charcoal
    name='True Signal'
))


# Trace 3: Mean Prediction
fig.add_trace(go.Scattergl(
    x=time_steps, y=y_mean_np[:, 0],
    mode='lines',
    line=dict(color='blue', width=2), # Solid, thick blue line
    name='Mean Prediction'
))

# Build the Dropdown Menu Logic
dropdown_buttons = []
for i in range(n_regions):
    # Calculate the new polygon shape for region i
    y_ci_new = np.concatenate([upper_np[:, i], lower_np[:, i][::-1]])
    
    button = dict(
        label=f'Region {i}',
        method='update',
        args=[
            # Update the 'y' arrays. Order must match the trace addition order: [CI, True, Sample, Mean]
            {
                'y': [y_ci_new, y_true_np[:, i], y_mean_np[:, i]]
            },
            {'title': f'Predicted vs Actual Data (Region {i})'}
        ]
    )
    dropdown_buttons.append(button)

# Final Layout Adjustments
fig.update_layout(
    title='Predicted vs Actual Data (Region 0)',
    xaxis_title='Time Steps',
    yaxis_title='Amplitude',
    updatemenus=[
        dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=0.01, xanchor='left',
            y=1.15, yanchor='top'
        )
    ],
    plot_bgcolor='white',
    xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) 
)

fig.show()