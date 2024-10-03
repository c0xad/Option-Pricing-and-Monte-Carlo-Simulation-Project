# variance_reduction.py

import numpy as np

class VarianceReductionTechnique:
    """
    Abstract base class for variance reduction techniques.
    """
    def apply(self, payoffs):
        raise NotImplementedError("Must implement apply method.")

class AntitheticVariates(VarianceReductionTechnique):
    """
    Antithetic Variates Technique.
    """
    def apply(self, payoffs_positive, payoffs_negative):
        return 0.5 * (payoffs_positive + payoffs_negative)

class ControlVariates(VarianceReductionTechnique):
    """
    Control Variates Technique.
    """
    def __init__(self, control_payoffs, control_price):
        self.control_payoffs = control_payoffs
        self.control_price = control_price
        self.beta = self.calculate_beta()

    def calculate_beta(self):
        covariance = np.cov(self.control_payoffs, self.control_payoffs)[0,1]
        variance = np.var(self.control_payoffs)
        return covariance / variance if variance != 0 else 0

    def apply(self, payoffs, option_price):
        adjusted_payoffs = payoffs + self.beta * (self.control_payoffs - self.control_price)
        return adjusted_payoffs

class ImportanceSampling(VarianceReductionTechnique):
    """
    Importance Sampling Technique.
    Adjusts the drift to focus sampling on important regions.
    """
    def __init__(self, shift):
        self.shift = shift
    
    def apply(self, Z):
        return Z + self.shift

    def weight(self, shift, Z):
        return np.exp(-shift * Z + 0.5 * shift ** 2)

    def __init__(self, delta):
        self.delta = delta  # Drift adjustment parameter

    def __init__(self, strata):
        self.strata = strata
    
    def apply(self, M):
        samples_per_stratum = M // self.strata
        samples = []
        for _ in range(self.strata):
            samples.append(np.random.uniform(0, 1, samples_per_stratum))
        return np.concatenate(samples)

    def apply(self, payoffs, S):
        # Example: Shift the mean of asset paths
        # This is a simplistic implementation; more sophisticated methods may be required
        adjusted_payoffs = payoffs * np.exp(-self.delta * S[:, -1])
        return adjusted_payoffs

class StratifiedSampling(VarianceReductionTechnique):
    """
    Stratified Sampling Technique.
    Divides the simulation space into strata and samples within each stratum.
    """
    def __init__(self, strata=10):
        self.strata = strata

    def apply(self, payoffs, S):
        M = len(payoffs)
        strata_size = M // self.strata
        adjusted_payoffs = np.zeros_like(payoffs)
        for i in range(self.strata):
            start = i * strata_size
            end = (i + 1) * strata_size if i != self.strata -1 else M
            # Sample uniformly within each stratum
            adjusted_payoffs[start:end] = np.mean(payoffs[start:end])
        return adjusted_payoffs
