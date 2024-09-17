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
    def __init__(self, process, option, T, N, M0, L, r, seed):
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

    def store_convergence_data(self, level, Y_l, V_l):
        self.convergence_data.append((level, Y_l, V_l))

    def mlmc_estimator(self):
        """
        Compute the MLMC estimator using adaptive sampling.

        Returns:
            price: Estimated option price.
            std_error: Standard error of the estimate.
        """
        Y_l = np.zeros(self.L + 1)
        V_l = np.zeros(self.L + 1)
        M_l = np.zeros(self.L + 1, dtype=int)

        for l in range(0, self.L + 1):
            M_l[l] = self.M0 * (2**l)
            batch_size = min(1000, M_l[l])  # Reduce batch size to 1000
            num_batches = M_l[l] // batch_size
            Y_l_sum = 0
            V_l_sum = 0

            for _ in range(num_batches):
                simulator = MonteCarloSimulator(
                    process=self.process,
                    option=self.option,
                    T=self.T,
                    N=self.N * (2**l),
                    M=batch_size,
                    r=self.r,
                    seed=self.seed
                )
                simulator.set_initial_price(self.S0)

                if l == 0:
                    price_l, _ = simulator.simulate(level=l)
                    Y_l_batch = price_l
                    V_l_batch = price_l**2
                else:
                    price_l, _ = simulator.simulate(level=l)
                    simulator_coarse = MonteCarloSimulator(
                        process=self.process,
                        option=self.option,
                        T=self.T,
                        N=self.N * (2**(l-1)),
                        M=batch_size,
                        r=self.r,
                        seed=self.seed
                    )
                    simulator_coarse.set_initial_price(self.S0)
                    price_l_minus_1, _ = simulator_coarse.simulate(level=l-1)
                    Y_l_batch = price_l - price_l_minus_1
                    V_l_batch = Y_l_batch**2

                Y_l_sum += np.sum(Y_l_batch)
                V_l_sum += np.sum(V_l_batch)

            Y_l[l] = Y_l_sum / M_l[l]
            V_l[l] = V_l_sum / M_l[l] - Y_l[l]**2

            print(f"Initial Sampling - Level {l}: Y_l = {Y_l[l]:.6f}, V_l = {V_l[l]:.6f}, M_l = {M_l[l]}")
            self.store_convergence_data(l, Y_l[l], V_l[l])

        mlmc_price = np.sum(Y_l)
        mlmc_variance = np.sum(V_l / M_l)
        std_error = np.sqrt(mlmc_variance)

        return mlmc_price, std_error

    def get_mlmc_convergence(self):
        return self.convergence_data
