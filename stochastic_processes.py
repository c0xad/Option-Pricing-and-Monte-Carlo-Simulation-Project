# stochastic_processes.py

import numpy as np

class StochasticProcess:
    """
    Abstract base class for stochastic processes.
    """
    def generate_paths(self, S0, T, N, M, level=0):
        raise NotImplementedError("Must implement generate_paths method.")

class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion model with support for varying levels.
    """
    def __init__(self, mu, sigma, r=0.0, seed=None):
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_1d(sigma)
        self.r = r
        self.seed = seed
        self.dim = len(self.mu)
        self.S0 = None

    def generate_paths(self, S0, T, N, M, level=0, Z=None):
        dt = T / N
        num_assets = len(S0) if isinstance(S0, (list, np.ndarray)) else 1
        
        if Z is None:
            Z = np.random.normal(0, 1, size=(M, N, num_assets))
        else:
            Z = Z.reshape(M, N, num_assets)
        
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        S = np.zeros((M, N + 1, num_assets))
        S[:, 0] = S0
        for t in range(1, N + 1):
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion[:, t-1])
        
        return S.squeeze()

class HestonModel(StochasticProcess):
    """
    Heston Stochastic Volatility Model with support for varying levels.
    """
    def __init__(self, S0, mu, sigma, kappa, theta, xi, rho, T, N, M, seed=None):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.kappa = kappa  # Rate of mean reversion
        self.theta = theta  # Long-term variance
        self.xi = xi        # Volatility of volatility
        self.rho = rho      # Correlation between the two Brownian motions
        self.T = T
        self.N = N
        self.M = M
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_paths(self):
        dt = self.T / self.N
        S = np.zeros((self.M, self.N + 1))
        V = np.zeros((self.M, self.N + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.sigma ** 2
        
        for t in range(1, self.N + 1):
            Z1 = np.random.standard_normal(self.M)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * np.random.standard_normal(self.M)
            V[:, t] = np.maximum(V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt +
                                 self.xi * np.sqrt(V[:, t-1] * dt) * Z1, 0)
            S[:, t] = S[:, t-1] * np.exp((self.mu - 0.5 * V[:, t-1]) * dt +
                                         np.sqrt(V[:, t-1] * dt) * Z2)
        return S

class JumpDiffusionProcess(StochasticProcess):
    def __init__(self, mu, sigma, lam, kappa, delta, r=0.0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.kappa = kappa
        self.delta = delta
        self.r = r
        self.seed = seed

    def generate_paths(self, S0, T, N, M, level=0, Z=None):
        dt = T / N
        np.random.seed(self.seed)
        size = (M, N)
        
        # Simulate Poisson jumps
        poisson_random = np.random.poisson(self.lam * dt, size)
        jump_sizes = np.random.normal(self.kappa, self.delta, size)
        jumps = poisson_random * (np.exp(jump_sizes) - 1)
        
        # Simulate GBM
        if Z is None:
            Z = np.random.standard_normal(size)
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        increments = drift + diffusion + np.log(1 + jumps)
        log_S = np.log(S0) + np.cumsum(increments, axis=1)
        S = np.exp(log_S)
        
        return np.column_stack((np.full(M, S0), S))
