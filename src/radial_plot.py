import numpy as np
import matplotlib.pyplot as plt
from src.Complex_radar import ComplexRadar

def radial_plot(dic):

    metrics = list(dic.keys())
    models = list(dic[metrics[0]].keys())

    data_matrix = {model: [np.mean(dic[metric][model]) for metric in metrics] for model in models}

    ranges = []
    for i, metric in enumerate(metrics):
        vals = [data_matrix[m][i] for m in models]
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            ranges.append((vmin - 0.1, vmax + 0.1))
        else:
            padding = (vmax - vmin) * 0.1
            ranges.append((vmin - padding, vmax + padding))

    fig = plt.figure(figsize=(6, 6))
    radar = ComplexRadar(fig, metrics, ranges)

    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple']

    for i, model in enumerate(models):
        radar.plot(data_matrix[model], label=model, color=colors[i], linewidth=2)
        radar.fill(data_matrix[model], facecolor=colors[i], alpha=0.15)

    radar.ax.set_title("Model Evaluation Metrics", pad=50)
    radar.use_legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(models))
    plt.show()

