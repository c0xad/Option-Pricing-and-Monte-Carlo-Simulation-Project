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
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.seed = seed

    def generate_paths(self, S0, T, N, M, level=0):
        """
        Generates asset price paths using Geometric Brownian Motion.
        
        Parameters:
            S0 (float or array): Initial asset price(s).
            T (float): Time to maturity.
            N (int): Number of time steps at the base level.
            M (int): Number of simulation paths.
            level (int): MLMC level, higher levels have finer discretization.
        
        Returns:
            S (array): Simulated asset price paths.
        """
        dt = T / (N * 2**level)
        num_steps = N * 2**level
        np.random.seed(self.seed)
        
        if isinstance(S0, (int, float)):
            S0 = [S0]
        
        num_assets = len(S0)
        Z = np.random.standard_normal((M, num_steps, num_assets))
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        S = np.zeros((M, num_steps + 1, num_assets))
        S[:, 0, :] = S0
        
        for i in range(num_steps):
            S[:, i+1, :] = S[:, i, :] * np.exp(drift + diffusion[:, i, :])
        
        if num_assets == 1:
            S = S.squeeze(axis=2)
        
        return S

class HestonModel(StochasticProcess):
    """
    Heston Stochastic Volatility Model with support for varying levels.
    """
    def __init__(self, mu, kappa, theta, sigma_v, rho, v0, r=0.0, seed=None):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.seed = seed

    def generate_paths(self, S0, T, N, M, level=0):
        """
        Generates asset price paths using the Heston model.
        
        Parameters:
            S0 (float or array): Initial asset price(s).
            T (float): Time to maturity.
            N (int): Number of time steps at the base level.
            M (int): Number of simulation paths.
            level (int): MLMC level, higher levels have finer discretization.
        
        Returns:
            S (array): Simulated asset price paths.
        """
        dt = T / (N * 2**level)
        num_steps = N * 2**level
        np.random.seed(self.seed)
        S = np.zeros((M, num_steps + 1))
        v = np.zeros((M, num_steps + 1))
        S[:, 0] = S0
        v[:, 0] = self.v0
        for t in range(1, num_steps + 1):
            Z1 = np.random.standard_normal(M)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(M)
            v_prev = v[:, t-1]
            v[:, t] = np.abs(v_prev + self.kappa * (self.theta - v_prev) * dt + 
                            self.sigma_v * np.sqrt(v_prev * dt) * Z1)
            S[:, t] = S[:, t-1] * np.exp((self.mu - 0.5 * v_prev) * dt + 
                                         np.sqrt(v_prev * dt) * Z2)
        return S
