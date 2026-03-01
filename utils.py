import numpy as np
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
    def __init__(self, Time_steps, tau, n, beta, gamma, delta_t):
        self.Time_steps = Time_steps
        self.beta = beta
        self.gamma = gamma
        self.delta_t = delta_t
        self.tau = tau
        self.n = n
        self.generate_data()

    def generate_data(self):
        delay_steps = int(self.tau / self.delta_t)
        total_steps = self.Time_steps + delay_steps + 1
        x_history = np.zeros(total_steps+1)
        x_history[:delay_steps] = 1.2 + np.random.uniform(-0.1, 0.1, delay_steps)

        for t in range(delay_steps, total_steps, 1):
            x_past = x_history[t - delay_steps]
            x_curr = x_history[t]

            x_succ = x_curr + ( self.beta * x_past / ( 1 + x_past**self.n )- self.gamma * x_curr ) * self.delta_t

            x_history[t + 1] = x_succ
        X = x_history[delay_steps+1:-1]
        X = (X - np.mean(X)) / np.std(X)
        Y = x_history[delay_steps+2:]
        Y = (Y - np.mean(Y)) / np.std(Y)
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