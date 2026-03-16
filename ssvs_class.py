import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from sklearn.metrics import mean_squared_error

# Ensure utils.py is in the same folder
from utils import zoomed_plots

class BayesianSSVS:
    def __init__(self):
        """
        Initializes the SSVS model class.
        The 'guide' and 'losses' will be populated during training.
        """
        self.guide = None
        self.losses = []

    def model(self, X, Y=None):
        """
        Bayesian Regression model with a Horseshoe prior to induce sparsity.
        """
        N_features = X.shape[1]
        
        # 1. Global Shrinkage (tau)
        tau = pyro.sample("tau", dist.HalfCauchy(1.0))
        
        # 2. Local Shrinkage (lambdas)
        with pyro.plate("features_plate", N_features):
            lambdas = pyro.sample("lambdas", dist.HalfCauchy(1.0))
            # Regression weights (beta)
            beta = pyro.sample("beta", dist.Normal(0.0, lambdas * tau))
            
        bias = pyro.sample("bias", dist.Normal(0.0, 10.0))
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        
        # X [1000, 500] @ beta [500] = mu [1000]
        mu = torch.matmul(X, beta) + bias
        
        # Remove the extra "1" dimension from Y (from [1000, 1] to [1000])
        if Y is not None:
            Y = Y.squeeze(-1)
        
        # Likelihood
        with pyro.plate("data_plate", X.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)

    def train(self, X_train, Y_train, epochs=2000, lr=0.01):
        """
        Trains the model using Stochastic Variational Inference (SVI).
        """
        pyro.clear_param_store()
        
        # Create the variational distribution (Guide) to approximate the Posterior
        self.guide = AutoDiagonalNormal(self.model)
        optimizer = Adam({"lr": lr})
        
        # Initialize the Stochastic Variational Inference
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        self.losses = []
        print(f"SSVS Training (via SVI) started for {epochs} epochs (LR={lr})...")
        for i in range(epochs):
            loss = svi.step(X_train, Y_train)
            self.losses.append(loss)
            if (i+1) % 500 == 0:
                print(f"Epoch {i+1}/{epochs} - ELBO Loss: {loss:.4f}")
        
        return self.guide, self.losses


    def evaluate(self, X_test, Y_test, num_samples=1000, target_name="Target"):
        """
        Generates probabilistic predictions and displays the plots.
        """
        if self.guide is None:
            raise ValueError("The model has not been trained yet! Run .train() before .evaluate().")

        # 1. Sampling using the guide
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs", "beta"))
        samples = predictive(X_test)
        y_pred_samples = samples["obs"]
        
        # 2. Median and Intervals Extraction
        y_pred_mean = torch.median(y_pred_samples, dim=0).values.detach().numpy()
        y_lower = torch.quantile(y_pred_samples, 0.025, dim=0).detach().numpy()
        y_upper = torch.quantile(y_pred_samples, 0.975, dim=0).detach().numpy()
        y_true = Y_test.detach().numpy()
        
        # Plots
        plt.figure(figsize=(12, 5))
        plt.plot(y_true, label="True Value", color="black", linewidth=1.5)
        plt.plot(y_pred_mean, label="Bayesian SSVS Median Forecast", color="blue", linestyle="-", linewidth=1.5)
        plt.fill_between(range(len(y_true)), y_lower.flatten(), y_upper.flatten(), color="blue", alpha=0.3, label=r"95\% Confidence Interval")
        plt.title(f"SSVS Probabilistic Forecast (Global) - {target_name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Target Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        

        beta_samples = samples["beta"].detach().numpy() 
        beta_mean = np.mean(beta_samples, axis=0)
        threshold = 0.05
        active_weights = np.sum(np.abs(beta_mean) > threshold)
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(beta_mean)), np.abs(beta_mean), color="green", alpha=0.7)
        plt.title(f"Sparsity of Reservoir Weights (SSVS)\nActive Weights: {active_weights} out of {len(beta_mean)}")
        plt.xlabel("Reservoir Node Index")
        plt.ylabel("Absolute Posterior Mean of Beta")
        plt.grid(True, alpha=0.3)
        plt.show()

        print("\nGenerating Zoomed Plots...")
        y_mean_2d = y_pred_mean.reshape(-1, 1)
        y_true_2d = y_true.reshape(-1, 1)
        y_upper_2d = y_upper.reshape(-1, 1)
        y_lower_2d = y_lower.reshape(-1, 1)
        
        # Make the zoomed plots
        zoomed_plots(
            region_idx=0, 
            y_mean=y_mean_2d, 
            y_true=y_true_2d, 
            upper=y_upper_2d, 
            lower=y_lower_2d, 
            steps=100, 
            n_plots=12, 
            rows=4
        )

    def compute_metrics(self, X_test, Y_test, num_samples=1000, taus=[0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]):
        """
        Computes the probabilistic metrics from the paper (MSE, Coverage, Width, Calibration, mCRPS).
        """
        if self.guide is None:
            raise ValueError("The model has not been trained yet! Run .train() before .compute_metrics().")

        # 1. Sampling
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs",))
        samples = predictive(X_test)
        y_pred_samples = samples["obs"].detach().numpy()
        y_true = Y_test.detach().numpy().flatten()
        
        # Flattening the samples along the features
        y_pred_samples = y_pred_samples.reshape(y_pred_samples.shape[0], -1)
        N = y_true.shape[0]
        y_preds = np.zeros((N, len(taus)))
        
        # Quantiles extraction
        for i, tau in enumerate(taus):
            y_preds[:, i] = np.quantile(y_pred_samples, tau, axis=0)
            
        # 2. MSE on the median
        mse = mean_squared_error(y_true, y_preds[:, taus.index(0.50)]) if 0.50 in taus else None
            
        # 3. Coverage and Width of the 95% interval
        if 0.025 in taus and 0.975 in taus:
            idx_lower = taus.index(0.025)
            idx_upper = taus.index(0.975)
            y_lower = y_preds[:, idx_lower]
            y_upper = y_preds[:, idx_upper]
            
            coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            width = np.mean(y_upper - y_lower)
        else:
            coverage, width = None, None
            
        # 4. Calibration Error
        cal_error = 0
        for i, tau in enumerate(taus):
            emp_tau = np.mean(y_true <= y_preds[:, i])
            cal_error += (tau - emp_tau)**2
            
        # 5. mCRPS (Approximated via the mean Pinball Loss over quantiles)
        crps_sum = 0
        for i, tau in enumerate(taus):
            r = y_true - y_preds[:, i]
            check_loss = np.where(r >= 0, tau * r, (tau - 1) * r)
            crps_sum += np.mean(check_loss)
        
        mcrps = (2 * crps_sum) / len(taus) 
        
        # --- FORMATTED PRINT ---
        print("\n" + "="*45)
        print("METRICS (Bayesian SSVS/SVI)")
        print("="*45)
        print(f"1. MSE (on median):           {mse:.4f}" if mse is not None else "1. MSE: N/A")
        print(f"2. Coverage (95% CI):         {coverage * 100:.2f}%" if coverage is not None else "2. Coverage: N/A")
        print(f"3. Mean Width (95% CI):       {width:.4f}" if width is not None else "3. Width: N/A")
        print(f"4. Calibration Error (cal):   {cal_error:.4f}")
        print(f"5. mCRPS (approximated):      {mcrps:.4f}")
        print("="*45 + "\n")
        
        return {
            "mse": mse,
            "coverage": coverage,
            "width": width,
            "calibration_error": cal_error,
            "mcrps": mcrps
        }