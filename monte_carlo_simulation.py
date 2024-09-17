# monte_carlo_simulation.py

import numpy as np
from joblib import Parallel, delayed
from variance_reduction import (
    AntitheticVariates, ControlVariates, ImportanceSampling, StratifiedSampling
)
from stochastic_processes import GeometricBrownianMotion, HestonModel
from option_models import AmericanOption

class MonteCarloSimulator:
    """
    Monte Carlo Simulator for option pricing.
    """
    def __init__(self, process, option, T, N, M, r=0.0, seed=None):
        self.process = process
        self.option = option
        self.T = T
        self.N = N
        self.M = M
        self.r = r
        self.seed = seed
        self.S0 = None

    def set_initial_price(self, S0):
        self.S0 = S0

    def simulate_gbm(self, level=0):
        return self.process.generate_paths(S0=self.S0, T=self.T, N=self.N, M=self.M, level=level)

    def simulate_heston(self, level=0):
        return self.process.generate_paths(S0=self.S0, T=self.T, N=self.N, M=self.M, level=level)

    def simulate(self, method=None, level=0):
        """
        Simulate option price using standard Monte Carlo methods.

        Parameters:
            method: Variance reduction technique (optional).
            level: Level of discretization for MLMC.

        Returns:
            price: Estimated option price.
            std_error: Standard error of the estimate.
        """
        if isinstance(self.process, GeometricBrownianMotion) or isinstance(self.process, HestonModel):
            S = self.process.generate_paths(S0=self.S0, T=self.T, N=self.N, M=self.M, level=level)
        else:
            raise NotImplementedError("Stochastic process not implemented.")

        if isinstance(self.option, AmericanOption):
            # Special handling for American Options
            price, std_error = self.option.price_with_lsm(self.process, self.S0, self.T, self.N, self.M, self.r, level=level)
            return price, std_error

        # Check if option is multi-asset
        if len(S.shape) == 3:
            payoffs = self.option.payoff(S)
        else:
            payoffs = self.option.payoff(S)

        discounted_payoffs = np.exp(-self.r * self.T) * payoffs

        if isinstance(method, AntitheticVariates):
            # Generate antithetic paths
            S_antithetic = self.process.generate_paths(S0=self.S0, T=self.T, N=self.N, M=self.M, level=level)
            if len(S_antithetic.shape) == 3 and len(S.shape) == 3:
                # Ensure same dimensionality for multi-asset
                S_antithetic = S_antithetic
            payoffs_antithetic = self.option.payoff(S_antithetic)
            discounted_payoffs_antithetic = np.exp(-self.r * self.T) * payoffs_antithetic
            # Apply Antithetic Variates
            adjusted_payoffs = method.apply(discounted_payoffs, discounted_payoffs_antithetic)
        elif isinstance(method, ControlVariates):
            # Control Variates require a control option with known price
            adjusted_payoffs = method.apply(payoffs, self.option.payoff(S))
            discounted_payoffs = np.exp(-self.r * self.T) * adjusted_payoffs
        elif isinstance(method, ImportanceSampling):
            # Apply Importance Sampling
            adjusted_payoffs = method.apply(payoffs, S)
            discounted_payoffs = np.exp(-self.r * self.T) * adjusted_payoffs
        elif isinstance(method, StratifiedSampling):
            # Apply Stratified Sampling
            adjusted_payoffs = method.apply(payoffs, S)
            discounted_payoffs = np.exp(-self.r * self.T) * adjusted_payoffs
        else:
            adjusted_payoffs = discounted_payoffs

        price = np.mean(adjusted_payoffs)
        std_error = np.std(adjusted_payoffs) / np.sqrt(self.M)
        return price, std_error

    def get_price_path(self):
        return self.process.generate_paths(self.S0, self.T, self.N, 1, level=0).flatten()

    def get_payoff_distribution(self):
        S = self.process.generate_paths(self.S0, self.T, self.N, self.M, level=0)
        return self.option.payoff(S)

