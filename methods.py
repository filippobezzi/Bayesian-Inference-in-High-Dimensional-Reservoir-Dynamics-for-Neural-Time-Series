import pyro
import torch
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA

from utils import evaluate_metrics, calibrate

def pyro_model(S, Y=None, print_shapes=False, target_dim=None, beta_params=[0,10.], scale=10.):
    """
    Args:
        S (Tensor/Array): Reservoir states of shape (rows, cols).
        Y (Tensor/Array, optional): Target of shape (rows, dim) or (rows,).
        beta_params (list): Normal prior [mean, std].
        scale (float): HalfNormal prior scale for sigma.
        target_dim (int, optional): Required if predicting (Y=None) with dim > 1.
    """
    if print_shapes: print(f"Tensor S shape: {S.shape}")

    rows, cols = S.shape
    S = torch.asarray(S, dtype=torch.float64)

    if Y is not None: 
        Y = torch.asarray(Y, dtype=torch.float64)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)  # Force shape (rows, 1)
        dim = Y.shape[1]
    else:
        # CRITICAL: Pass `target_dim` so the model knows what shape of `beta` to sample.
        dim = target_dim if target_dim is not None else 1

    if print_shapes and Y is not None: 
        print(f"Tensor Y shape: {Y.shape}")

    # Beta Parameter Shape: (dim, cols)
    # .to_event(2) -> both dimensions part of the same global parameter sample.
    beta = pyro.sample(
        "beta",
        pyro.distributions.Normal(
            torch.tensor(beta_params[0], dtype=torch.float64),
            torch.tensor(beta_params[1], dtype=torch.float64)
        ).expand([dim, cols]).to_event(2)
    )
    
    sigma = pyro.sample(
        "sigma",
        pyro.distributions.HalfNormal(torch.tensor(scale, dtype=torch.float64))
    )

    # (rows, cols) @ (cols, dim) -> (rows, dim)
    mu = torch.matmul(S, beta.mT)

    with pyro.plate("data", size=rows, dim=-1): # Enforce assignment of Target dim to 
        while sigma.ndim < mu.ndim:
            sigma = sigma.unsqueeze(-1)

        if print_shapes:
            print(f"Tensor Mean of Posterior shape: {mu.shape, mu.ndim}")
            print(f"Tensor Std of Posterior shape: {sigma.shape, sigma.ndim}")
            if Y is not None: 
                print(f"Tensor Y of Posterior shape: {Y.shape, Y.ndim}")
                
        # Sampling
        # .to_event(1) -> (`dim`) single multivariate event / 
        # batch shape -> (rows,), which matches the plate length.
        pyro.sample(
            "obs",
            pyro.distributions.Normal(mu, sigma).to_event(1),
            obs=Y
        )


class Reservoir:
    def __init__(self, input_dim: int, reservoir_dim: int, spectral_radius: float, seed=None):
        """_summary_

        Args:
            input_dim (_type_): _description_
            reservoir_dim (_type_): _description_
            spectral_radius (_type_): _description_
            seed (_type_, optional): _description_. Defaults to None.
        """
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.w_init()

    def w_init( self ):
        """_summary_
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        W = np.random.uniform( -1, 1, (self.reservoir_dim, self.reservoir_dim) ) 
        eigvals_res = np.linalg.eigvals( W )
        max_radius = np.max( np.abs( eigvals_res ) ) 
        self.W = W * self.spectral_radius / max_radius
        self.W_in = np.random.uniform( -1, 1, (self.reservoir_dim, self.input_dim) )

    def get_states(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        Time_steps = X.shape[0]
        states = np.zeros( (Time_steps, self.reservoir_dim) )
        s_prev = np.zeros( (1, self.reservoir_dim) )

        for t in range(Time_steps):
            x_t = X[t, :]
            #if t==0: print(self.W_in.shape, x_t.shape, s_prev.shape, self.W.shape)  # debug
            s_curr = np.tanh(self.W_in @ x_t + s_prev @ self.W )
            states[t,:] = s_curr
            s_prev = s_curr

        return states

import torch
import pyro
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from methods import pyro_model, Reservoir
from utils import evaluate_metrics, calibrate

class BayesianStateMCMC(BaseEstimator, RegressorMixin):
    """
    Streamlined MCMC Estimator designed to ingest pre-computed, synchronized Reservoir States.
    """
    def __init__(self, beta_params=(0, 1.0), scale=1.0, n_components=64, 
                 num_samples=200, warmup_steps=200, seed=42):
        self.beta_params = beta_params
        self.scale = scale
        self.n_components = n_components
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.seed = seed

    def fit(self, S_blocks, y_blocks):
        # Stack blocks for continuous PCA and training
        S_train = np.vstack(S_blocks)
        y_train = np.vstack(y_blocks)

        # PCA Reduction on States
        self.pca_ = PCA(n_components=self.n_components)
        S_train_PCA = self.pca_.fit_transform(S_train)

        S_tensor = torch.tensor(S_train_PCA, dtype=torch.float64)
        y_tensor = torch.tensor(y_train, dtype=torch.float64).squeeze()

        # MCMC Inference
        nut_kernel = pyro.infer.NUTS(pyro_model)
        self.mcmc_ = pyro.infer.MCMC(nut_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)
        
        # Explicit kwargs to avoid positional dimension collapse
        self.mcmc_.run(
            S=S_tensor, 
            Y=y_tensor, 
            beta_params=self.beta_params, 
            scale=self.scale, 
            print_shapes=False, 
            target_dim=y_train.shape[1]
        )
        
        self.posterior_samples_ = self.mcmc_.get_samples()
        self.target_dim_ = y_train.shape[1]
        return self

    def predict(self, S_blocks):
        S_test = np.vstack(S_blocks)
        S_test_PCA = self.pca_.transform(S_test)
        S_tensor = torch.tensor(S_test_PCA, dtype=torch.float64)

        predictive = pyro.infer.Predictive(pyro_model, posterior_samples=self.posterior_samples_, num_samples=None, parallel=True)
        
        # Extract distributional predictions
        self.y_pred_samples_z_ = predictive(
            S=S_tensor, 
            Y=None, 
            beta_params=self.beta_params, 
            scale=self.scale, 
            print_shapes=False, 
            target_dim=self.target_dim_
        )["obs"]

        # Sklearn compatibility (mean point estimate)
        return self.y_pred_samples_z_.mean(dim=0).detach().cpu().numpy()

    
class DataGenerator:
    def __init__(self, Time_steps, tau, n=10, beta=0.2, gamma=0.1, delta_t=0.1, init_steps=500 , under_samp=10):
        """
        Method to generates a caothic time-series using the Mackay-Glass Equation according to Eq.(9) in https://doi.org/10.1016/j.mlwa.2022.100300

        Args:
            Time_steps (int): The length of the time-series
            tau (float): Delay parameter (caothic behavior for tau >= 17)
            n (int, optional): Constant parameter. Defaults to 10.
            beta (float, optional): Constant parameter. Defaults to 0.2.
            gamma (float, optional): Constant parameter. Defaults to 0.1.
            delta_t (float, optional): integration step size
            init_steps (int, optional): Excluded steps for caothic behavior to emerge (heuristic). Defaults to 500.
            under_samp (int, optional): Undersampling to increase variability (heuristic). Defaults to 10.
        """
        self.Time_steps = Time_steps*under_samp
        self.under_samp = under_samp
        self.init_steps = init_steps
        self.beta = beta
        self.gamma = gamma
        self.delta_t = delta_t
        self.tau = tau
        self.n = n

    def generate_data(self):
        """
        It generates the time-series

        Returns:
            X [Time_steps, 1]: Input time-series
            Y [Time_steps, 1]: Output time-series
        """
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
        # X = (X - np.mean(X)) / ( np.std(X) + 1e-8 )
        Y = x_history[skip_steps+1:][::self.under_samp]
        # Y = (Y - np.mean(Y)) / ( np.std(Y) + 1e-8 )

        return X.reshape((-1, 1)), Y.reshape((-1, 1))