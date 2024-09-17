# quasi_random.py

from scipy.stats import qmc
import numpy as np

def generate_sobol_samples(d, n, seed=None):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    return qmc.scale(sampler.random(n=n), -5, 5)

def generate_sobol_paths(N, M, dim, seed=None):
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    samples = sampler.random(n=N*M)
    samples = qmc.scale(samples, -5, 5)
    return samples.reshape(M, N, dim)