from utils.ESNDataset import ESNDataset
from utils.ESNVariational import ESNVariational
import torch
import matplotlib.pyplot as plt


def prediction_curve(esnvariational_object: ESNVariational, test_dataset: ESNDataset, scaler: dict, title = "Predicted vs actual data"):
    
    y_samples = esnvariational_object.predict(test_states=test_dataset.states)
    y_true = test_dataset.normalized_predictions
    
    if scaler:
        m, s = scaler['mean'], scaler['std']
        y_samples = (y_samples * s) + m
        y_true = (y_true * s) + m

    mean_pred = y_samples.median(dim=0).values
    lower_bound = torch.quantile(y_samples, 0.025, dim=0) # 95% CI Lower
    upper_bound = torch.quantile(y_samples, 0.975, dim=0) # 95% CI Upper

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual Load", color="black", alpha=0.6)
    plt.plot(mean_pred, label="Bayesian Mean", color="blue")
    
    plt.fill_between(
        range(len(mean_pred)), 
        lower_bound, 
        upper_bound, 
        color="blue", alpha=0.2, label="95% Confidence"
    )
    
    plt.title(title)
    plt.legend()
    plt.show()

    return y_true, y_samples, mean_pred, lower_bound, upper_bound


