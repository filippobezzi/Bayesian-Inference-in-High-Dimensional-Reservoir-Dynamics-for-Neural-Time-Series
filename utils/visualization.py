import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_functional_connectivity(x_true, x_pred):
    fig, ax = plt.subplots(
        1, 3, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1, 0.05]}
    )
    fig.suptitle("Functional connectivity", fontsize=16)

    ############ PLOT 1 ###########
    ax[0].set_title("True")
    # Set the labels
    corr = np.corrcoef(x_true, rowvar=False)  # regions x regions
    sns.heatmap(
        corr, vmin=-1, vmax=1, cmap="vlag", ax=ax[0], cbar=False
    )  # , cbar_kws={'label':'Pearson r'})
    ax[0].set_xlabel("Regions", fontsize=16)
    ax[0].set_ylabel("Regions", fontsize=16)

    # PLOT 2

    corr_rec = np.corrcoef(x_pred, rowvar=False)  # regions x regions
    sns.heatmap(
        corr_rec, vmin=-1, vmax=1, cmap="vlag", ax=ax[1], cbar=True, cbar_ax=ax[2]
    )  # , cbar_kws={'label':'Pearson r'})
    # Set the labels
    ax[1].set_title("Predicted")
    ax[1].set_xlabel("Regions", fontsize=16)
    # ax[1].set_ylabel("Regions", fontsize=16)
    plt.show()
    return
