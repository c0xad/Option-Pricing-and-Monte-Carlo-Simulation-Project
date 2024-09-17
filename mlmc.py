# mlmc_simulation.py (Updated)

import numpy as np
from joblib import Parallel, delayed
from monte_carlo_simulation import MonteCarloSimulator
from variance_reduction import AntitheticVariates, ControlVariates
from quasi_random import generate_sobol_paths
from stochastic_processes import GeometricBrownianMotion, HestonModel
from option_models import EuropeanCallOption
import matplotlib.pyplot as plt

class MLMC_Simulator:
    def __init__(self, process, option, T, N, M0, L, r, seed, num_cores=None):
        self.process = process
        self.option = option
        self.T = T
        self.N = N
        self.M0 = M0
        self.L = L
        self.r = r
        self.seed = seed
        self.num_cores = num_cores
        self.S0 = None
        self.batch_size = 1000
        self.convergence_data = []

    def set_initial_price(self, S0):
        self.S0 = S0

    def mlmc_estimator(self, epsilon=0.0001, max_iterations=100):
        Y_l = np.zeros(self.L + 1)
        V_l = np.zeros(self.L + 1)
        M_l = np.zeros(self.L + 1, dtype=int)
        C_l = np.array([self.compute_cost(l) for l in range(self.L + 1)])

        for iteration in range(max_iterations):
            for l in range(0, self.L + 1):
                if iteration == 0:
                    M_l[l] = self.M0
                else:
                    M_l_optimal = self.compute_optimal_M_l(V_l, C_l, epsilon)
                    M_l[l] = max(M_l[l], M_l_optimal[l])

                Y_l_new, V_l_new = self.simulate_level(l, M_l[l])
                
                if iteration == 0:
                    Y_l[l] = Y_l_new
                    V_l[l] = V_l_new
                else:
                    Y_l[l] = (Y_l[l] * iteration + Y_l_new) / (iteration + 1)
                    V_l[l] = (V_l[l] * iteration + V_l_new) / (iteration + 1)

                print(f"Iteration {iteration}, Level {l}: Y_l = {Y_l[l]:.6f}, V_l = {V_l[l]:.6f}, M_l = {M_l[l]}")

            mlmc_price = np.sum(Y_l)
            mlmc_variance = np.sum(V_l / M_l)
            std_error = np.sqrt(mlmc_variance)
            
            self.convergence_data.append((iteration, mlmc_price, std_error))
            print(f"Iteration {iteration}: MLMC Price = {mlmc_price:.6f}, Std Error = {std_error:.6f}")
            
            if std_error < epsilon and iteration > 10:
                break

        return mlmc_price, std_error

    def simulate_level(self, l, M):
        batch_size = min(self.batch_size, M)
        num_batches = M // batch_size

        def simulate_batch(batch_index):
            seed = self.seed + l * 1000 + batch_index
            N_fine = self.N * (2**l)
            N_coarse = N_fine // 2 if l > 0 else N_fine
            Z = generate_sobol_paths(N_fine, batch_size, self.process.dim, seed=seed)

            simulator_fine = MonteCarloSimulator(
                process=self.process,
                option=self.option,
                T=self.T,
                N=N_fine,
                M=batch_size,
                r=self.r,
                seed=seed
            )
            simulator_fine.set_initial_price(self.S0)
            price_fine, _ = simulator_fine.simulate(level=l, Z=Z)

            if l == 0:
                Y_l_batch = price_fine
                V_l_batch = price_fine**2
            else:
                Z_coarse = Z[:, ::2]  # Use every other point for the coarse level
                simulator_coarse = MonteCarloSimulator(
                    process=self.process,
                    option=self.option,
                    T=self.T,
                    N=N_coarse,
                    M=batch_size,
                    r=self.r,
                    seed=seed
                )
                simulator_coarse.set_initial_price(self.S0)
                price_coarse, _ = simulator_coarse.simulate(level=l-1, Z=Z_coarse)
                Y_l_batch = price_fine - price_coarse
                V_l_batch = Y_l_batch**2

            return np.sum(Y_l_batch), np.sum(V_l_batch)

        results = Parallel(n_jobs=self.num_cores)(delayed(simulate_batch)(i) for i in range(num_batches))
        Y_l_sum, V_l_sum = np.sum(results, axis=0)

        Y_l = Y_l_sum / M
        V_l = V_l_sum / M - Y_l**2

        return Y_l, V_l

    def compute_optimal_M_l(self, V_l, C_l, epsilon):
        sum_sqrt_VC = np.sum(np.sqrt(V_l * C_l))
        M_l_optimal = np.ceil((2 / epsilon**2) * sum_sqrt_VC * np.sqrt(V_l / C_l)).astype(int)
        return M_l_optimal

    def compute_cost(self, l):
        return self.N * (2**l)

    def get_mlmc_convergence(self):
        return self.convergence_data
