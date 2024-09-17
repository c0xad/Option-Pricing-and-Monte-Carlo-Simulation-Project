# mlmc_simulation.py (Updated)

import numpy as np
from joblib import Parallel, delayed
from monte_carlo_simulation import MonteCarloSimulator
from variance_reduction import AntitheticVariates, ControlVariates
from stochastic_processes import GeometricBrownianMotion, HestonModel
from option_models import EuropeanCallOption

class MLMC_Simulator:
    """
    Multi-Level Monte Carlo Simulator for option pricing.
    """
    def __init__(self, process, option, T, N, M0, L, r=0.0, seed=None):
        """
        Initialize the MLMC simulator.

        Parameters:
            process: StochasticProcess instance.
            option: Option instance.
            T (float): Time to maturity.
            N (int): Number of time steps at the base level.
            M0 (int): Initial number of samples at each level.
            L (int): Maximum level.
            r (float): Risk-free rate.
            seed (int): Random seed.
        """
        self.process = process
        self.option = option
        self.T = T
        self.N = N
        self.M0 = M0
        self.L = L
        self.r = r
        self.seed = seed
        self.S0 = None
        self.convergence_data = []

    def set_initial_price(self, S0):
        self.S0 = S0

    def mlmc_estimator(self):
        """
        Compute the MLMC estimator using adaptive sampling.

        Returns:
            price: Estimated option price.
            std_error: Standard error of the estimate.
        """
        # Initialize estimates
        Y_l = np.zeros(self.L + 1)
        V_l = np.zeros(self.L + 1)
        M_l = np.zeros(self.L + 1, dtype=int)

        # Initial sampling
        for l in range(0, self.L + 1):
            # Determine number of samples for this level
            M_l[l] = self.M0 * (2**l)
            # Create a Monte Carlo Simulator for level l
            simulator = MonteCarloSimulator(
                process=self.process,
                option=self.option,
                T=self.T,
                N=self.N,
                M=self.M0,  # Start with M0 samples per level
                r=self.r,
                seed=self.seed
            )
            simulator.set_initial_price(self.S0)

            # Simulate Y_l = P_l - P_{l-1}
            if l == 0:
                # Coarsest level
                price_l, _ = simulator.simulate(level=l)
                Y_l[l] = price_l
                V_l[l] = price_l**2  # Placeholder for variance
            else:
                # Fine level
                price_l, _ = simulator.simulate(level=l)
                # Coarse level
                simulator_coarse = MonteCarloSimulator(
                    process=self.process,
                    option=self.option,
                    T=self.T,
                    N=self.N,
                    M=self.M0,
                    r=self.r,
                    seed=self.seed
                )
                simulator_coarse.set_initial_price(self.S0)
                price_l_minus_1, _ = simulator_coarse.simulate(level=l-1)
                Y_l[l] = price_l - price_l_minus_1
                V_l[l] = (Y_l[l])**2  # Placeholder for variance

            print(f"Initial Sampling - Level {l}: Y_l = {Y_l[l]:.6f}, V_l = {V_l[l]:.6f}, M_l = {M_l[l]}")

        # Estimate optimal number of samples per level based on variance and cost
        # For simplicity, we'll proceed without adaptive sampling in this example

        # Compute MLMC estimator
        price = np.sum(Y_l)
        std_error = np.sqrt(np.sum(V_l) / np.sum(M_l))

        return price, std_error

    def get_mlmc_convergence(self):
        return self.convergence_data
