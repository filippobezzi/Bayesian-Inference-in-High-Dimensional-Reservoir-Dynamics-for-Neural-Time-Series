import numpy as np

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