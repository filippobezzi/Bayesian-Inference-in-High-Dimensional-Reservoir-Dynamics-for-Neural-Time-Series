import torch
import torch.nn as nn

class Reservoir(nn.Module):
    def __init__(self, N, K, spectral_radius=0.9):
        super(Reservoir, self).__init__()
        
        self.N = N  # Number of reservoir neurons
        self.K = K  # Input dimension

        
        # Initialize Input Matrix (Uniform distribution is common)
        W_in = torch.rand(N, K) * 2 - 1  # Range [-1, 1]
        
        # Initialize Reservoir Matrix (Gaussian)
        W = torch.randn(N, N)
        
        # Scale Spectral Radius for stability
        # We calculate the largest eigenvalue and scale W
        with torch.no_grad():
            # Use real/imaginary parts to find the magnitude of eigenvalues
            eigenvalues = torch.linalg.eigvals(W)
            max_eig = torch.max(torch.abs(eigenvalues))
            W = W * (spectral_radius / max_eig)
        
        # Register as buffers (Fixed weights, move with model to GPU)
        self.register_buffer('W_in', W_in)
        self.register_buffer('W', W)
        
        # Internal state (Initialized to zeros)
        self.register_buffer('states', torch.zeros(1, N))
        
        self.activation = nn.ReLU()

    def forward(self, x, y = None):
        """
        x shape: [Batch, K]
        Returns the updated state: [Batch, N]
        """
        
        # Linear combinations: Input effect + Reservoir recurrent effect
        # We use .t() on weights because x is [Batch, K] and states is [Batch, N]
        input_part = x @ self.W_in.t()
        recurrent_part = self.states @ self.W.t()
        
        # We .detach() to ensure we don't track gradients through time steps
        self.states = self.activation(input_part + recurrent_part).detach()
        
        return self.states

    def reset_state(self, batch_size=1):
        """Clears the memory of the reservoir."""
        self.states = torch.zeros(batch_size, self.N, device=self.W.device)