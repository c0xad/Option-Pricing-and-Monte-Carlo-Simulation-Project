# option_models.py

import numpy as np

class Option:
    """
    Abstract base class for options.
    """
    def payoff(self, S):
        raise NotImplementedError("Must implement payoff method.")
    
    def greek_delta(self, S, epsilon=1e-4):
        """
        Estimate Delta using finite differences.
        """
        raise NotImplementedError("Must implement greek_delta method.")
    
    def greek_gamma(self, S, epsilon=1e-4):
        """
        Estimate Gamma using finite differences.
        """
        raise NotImplementedError("Must implement greek_gamma method.")
    
    def greek_vega(self, process, option_price, d_sigma=1e-4):
        """
        Estimate Vega by bumping the volatility.
        """
        raise NotImplementedError("Must implement greek_vega method.")


class EuropeanCallOption(Option):
    """
    European Call Option.
    """
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(S[:, -1] - self.K, 0)

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = np.maximum(S_up[:, -1] - self.K, 0)
        payoff_down = np.maximum(S_down[:, -1] - self.K, 0)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = np.maximum(S_up[:, -1] - self.K, 0)
        payoff_middle = np.maximum(S[:, -1] - self.K, 0)
        payoff_down = np.maximum(S_down[:, -1] - self.K, 0)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class EuropeanPutOption(Option):
    """
    European Put Option.
    """
    def __init__(self, K):
        self.K = K

    def payoff(self, S):
        return np.maximum(self.K - S[:, -1], 0)

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = np.maximum(self.K - S_up[:, -1], 0)
        payoff_down = np.maximum(self.K - S_down[:, -1], 0)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = np.maximum(self.K - S_up[:, -1], 0)
        payoff_middle = np.maximum(self.K - S[:, -1], 0)
        payoff_down = np.maximum(self.K - S_down[:, -1], 0)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class AsianCallOption(Option):
    """
    Asian Call Option.
    """
    def __init__(self, K, averaging='arithmetic'):
        self.K = K
        self.averaging = averaging.lower()

    def payoff(self, S):
        if self.averaging == 'arithmetic':
            average_S = np.mean(S, axis=1)
        elif self.averaging == 'geometric':
            average_S = np.exp(np.mean(np.log(S), axis=1))
        else:
            raise ValueError("Averaging method must be 'arithmetic' or 'geometric'.")
        return np.maximum(average_S - self.K, 0)

    def greek_delta(self, S, epsilon=1e-4):
        # Similar to EuropeanCallOption
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        if self.averaging == 'arithmetic':
            average_up = np.mean(S_up, axis=1)
            average_down = np.mean(S_down, axis=1)
        else:
            average_up = np.exp(np.mean(np.log(S_up), axis=1))
            average_down = np.exp(np.mean(np.log(S_down), axis=1))
        payoff_up = np.maximum(average_up - self.K, 0)
        payoff_down = np.maximum(average_down - self.K, 0)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Similar to EuropeanCallOption
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        if self.averaging == 'arithmetic':
            average_up = np.mean(S_up, axis=1)
            average_middle = np.mean(S, axis=1)
            average_down = np.mean(S_down, axis=1)
        else:
            average_up = np.exp(np.mean(np.log(S_up), axis=1))
            average_middle = np.exp(np.mean(np.log(S), axis=1))
            average_down = np.exp(np.mean(np.log(S_down), axis=1))
        payoff_up = np.maximum(average_up - self.K, 0)
        payoff_middle = np.maximum(average_middle - self.K, 0)
        payoff_down = np.maximum(average_down - self.K, 0)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Similar to EuropeanCallOption
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        if self.averaging == 'arithmetic':
            average_S = np.mean(S, axis=1)
        else:
            average_S = np.exp(np.mean(np.log(S), axis=1))
        payoff_up = np.maximum(average_S - self.K, 0)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class BarrierOption(Option):
    """
    Barrier Option (Up-and-Out, Down-and-Out, Up-and-In, Down-and-In).
    """
    def __init__(self, K, barrier, option_type='up-and-out', rebate=0.0):
        self.K = K
        self.barrier = barrier
        self.option_type = option_type.lower()
        self.rebate = rebate

    def payoff(self, S):
        if self.option_type == 'up-and-out':
            knocked_out = np.any(S >= self.barrier, axis=1)
            return np.where(knocked_out, self.rebate, np.maximum(S[:, -1] - self.K, 0))
        elif self.option_type == 'down-and-out':
            knocked_out = np.any(S <= self.barrier, axis=1)
            return np.where(knocked_out, self.rebate, np.maximum(S[:, -1] - self.K, 0))
        elif self.option_type == 'up-and-in':
            knocked_in = np.any(S >= self.barrier, axis=1)
            return np.where(knocked_in, np.maximum(S[:, -1] - self.K, 0), self.rebate)
        elif self.option_type == 'down-and-in':
            knocked_in = np.any(S <= self.barrier, axis=1)
            return np.where(knocked_in, np.maximum(S[:, -1] - self.K, 0), self.rebate)
        else:
            raise ValueError("Unsupported barrier option type. Use 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'.")

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_down = self.payoff(S_down)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_middle = self.payoff(S)
        payoff_down = self.payoff(S_down)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class LookbackOption(Option):
    """
    Lookback Option (Floating Strike or Fixed Strike).
    """
    def __init__(self, K=None, option_type='floating', is_call=True):
        """
        :param K: Strike price for fixed strike Lookback Option. Ignored if floating.
        :param option_type: 'floating' or 'fixed'.
        :param is_call: True for Call, False for Put.
        """
        self.K = K
        self.option_type = option_type.lower()
        self.is_call = is_call

    def payoff(self, S):
        if self.option_type == 'floating':
            if self.is_call:
                # Floating Strike Lookback Call: S_T - min(S)
                min_S = np.min(S, axis=1)
                return S[:, -1] - min_S
            else:
                # Floating Strike Lookback Put: max(S) - S_T
                max_S = np.max(S, axis=1)
                return max_S - S[:, -1]
        elif self.option_type == 'fixed':
            if self.K is None:
                raise ValueError("Strike price K must be provided for fixed strike Lookback Option.")
            if self.is_call:
                # Fixed Strike Lookback Call: max(S) - K
                max_S = np.max(S, axis=1)
                return np.maximum(max_S - self.K, 0)
            else:
                # Fixed Strike Lookback Put: K - min(S)
                min_S = np.min(S, axis=1)
                return np.maximum(self.K - min_S, 0)
        else:
            raise ValueError("Unsupported Lookback option type. Use 'floating' or 'fixed'.")

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_down = self.payoff(S_down)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_middle = self.payoff(S)
        payoff_down = self.payoff(S_down)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class AmericanOption(Option):
    """
    American Option using Least-Squares Monte Carlo (LSM) for pricing.
    """
    def __init__(self, K, is_call=True):
        self.K = K
        self.is_call = is_call

    def payoff(self, S):
        if self.is_call:
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def price_with_lsm(self, process, S0, T, N, M, r, level=0):
        """
        Price American option using Least-Squares Monte Carlo (Longstaff-Schwartz method).
        """
        dt = T / (N * 2**level)
        discount_factor = np.exp(-r * dt)
        S = process.generate_paths(S0=S0, T=T, N=N, M=M, level=level)
        payoff = self.payoff(S)
        # Initialize cash flows
        cash_flows = payoff[:, -1]
        for t in range(N-1, 0, -1):
            in_the_money = (self.payoff(S[:, t:t+1]) > 0).flatten()
            if not np.any(in_the_money):
                continue
            X = S[in_the_money, t]
            Y = cash_flows[in_the_money] * discount_factor
            # Basis functions: polynomial (e.g., 1, S, S^2)
            A = np.vstack([np.ones_like(X), X, X**2]).T
            # Regression to find continuation value
            coeff = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation_value = coeff[0] + coeff[1]*X + coeff[2]*X**2
            # Immediate exercise value
            immediate_exercise = self.payoff(S[in_the_money, t:t+1]).flatten()
            # Decide whether to exercise
            exercise = immediate_exercise > continuation_value
            cash_flows[in_the_money] = np.where(exercise, immediate_exercise, cash_flows[in_the_money] * discount_factor)
        # Discount to present
        price = np.mean(cash_flows * np.exp(-r * T))
        std_error = np.std(cash_flows * np.exp(-r * T)) / np.sqrt(M)
        return price, std_error

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        # Note: American options are path-dependent; Delta estimation is more complex
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_down = self.payoff(S_down)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_middle = self.payoff(S)
        payoff_down = self.payoff(S_down)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        # Note: For American options, pricing with LSM is computationally intensive
        # Here, we provide a simplified estimation
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class RainbowOption(Option):
    """
    Rainbow Option based on multiple underlying assets.
    """
    def __init__(self, K, weights=None, max_min='max'):
        """
        :param K: Strike price
        :param weights: List of weights for each underlying asset
        :param max_min: 'max' for best-of, 'min' for worst-of
        """
        self.K = K
        self.weights = weights  # e.g., [0.5, 0.5] for two assets
        self.max_min = max_min.lower()

    def payoff(self, S):
        """
        S: array of shape (M, N+1, d) where d is the number of assets
        """
        if len(S.shape) == 2:
            raise ValueError("For Rainbow Options, S should have three dimensions: (M, N+1, d)")
        if self.weights is None:
            weights = np.ones(S.shape[2]) / S.shape[2]
        else:
            weights = np.array(self.weights)
            if len(weights) != S.shape[2]:
                raise ValueError("Number of weights must match number of underlying assets.")
        if self.max_min == 'max':
            # Best-of option: max(weight_i * S_i)
            weighted_S = S[:, -1, :] * weights
            payoff = np.max(weighted_S, axis=1) - self.K
        elif self.max_min == 'min':
            # Worst-of option: min(weight_i * S_i)
            weighted_S = S[:, -1, :] * weights
            payoff = self.K - np.min(weighted_S, axis=1)
        else:
            raise ValueError("max_min must be 'max' or 'min'.")
        if self.max_min == 'max':
            return np.maximum(payoff, 0)
        else:
            return np.maximum(payoff, 0)

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        # Note: For multi-asset options, Delta is a vector
        deltas = []
        for asset in range(S.shape[2]):
            S_up = S.copy()
            S_down = S.copy()
            S_up[:, 0, asset] += epsilon
            S_down[:, 0, asset] -= epsilon
            payoff_up = self.payoff(S_up)
            payoff_down = self.payoff(S_down)
            delta = (payoff_up - payoff_down) / (2 * epsilon)
            deltas.append(delta)
        return np.array(deltas)  # Shape: (d, M)

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        deltas = self.greek_delta(S, epsilon)
        S_up = S.copy()
        S_down = S.copy()
        for asset in range(S.shape[2]):
            S_up[:, 0, asset] += epsilon
            S_down[:, 0, asset] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_down = self.payoff(S_down)
        gamma = (payoff_up - 2 * self.payoff(S) + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma for all assets
        # Assuming process handles multiple assets
        original_sigma = process.sigma.copy()
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class SpreadOption(Option):
    """
    Spread Option on two underlying assets.
    """
    def __init__(self, K, asset_index_1=0, asset_index_2=1, is_call=True):
        """
        :param K: Strike price
        :param asset_index_1: Index of the first asset in S
        :param asset_index_2: Index of the second asset in S
        :param is_call: True for Call on spread, False for Put on spread
        """
        self.K = K
        self.asset_index_1 = asset_index_1
        self.asset_index_2 = asset_index_2
        self.is_call = is_call

    def payoff(self, S):
        """
        S: array of shape (M, N+1, d) where d >= 2
        """
        spread = S[:, -1, self.asset_index_1] - S[:, -1, self.asset_index_2]
        if self.is_call:
            return np.maximum(spread - self.K, 0)
        else:
            return np.maximum(self.K - spread, 0)

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        deltas = []
        for asset in range(S.shape[2]):
            S_up = S.copy()
            S_down = S.copy()
            S_up[:, 0, asset] += epsilon
            S_down[:, 0, asset] -= epsilon
            payoff_up = self.payoff(S_up)
            payoff_down = self.payoff(S_down)
            delta = (payoff_up - payoff_down) / (2 * epsilon)
            deltas.append(delta)
        return np.array(deltas)  # Shape: (d, M)

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        deltas_up = self.greek_delta(S + epsilon, epsilon)
        deltas_down = self.greek_delta(S - epsilon, epsilon)
        gamma = (deltas_up - deltas_down) / (2 * epsilon)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma for all assets
        original_sigma = process.sigma.copy()
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class DigitalOption(Option):
    """
    Digital (Binary) Option.
    """
    def __init__(self, K, payout=1.0, is_call=True):
        """
        :param K: Strike price
        :param payout: Payout if option is in the money
        :param is_call: True for Call, False for Put
        """
        self.K = K
        self.payout = payout
        self.is_call = is_call

    def payoff(self, S):
        if self.is_call:
            return np.where(S[:, -1] > self.K, self.payout, 0.0)
        else:
            return np.where(S[:, -1] < self.K, self.payout, 0.0)

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_down = self.payoff(S_down)
        delta = (payoff_up - payoff_down) / (2 * epsilon)
        return delta

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        S_up = S.copy()
        S_down = S.copy()
        S_up[:, 0] += epsilon
        S_down[:, 0] -= epsilon
        payoff_up = self.payoff(S_up)
        payoff_middle = self.payoff(S)
        payoff_down = self.payoff(S_down)
        gamma = (payoff_up - 2 * payoff_middle + payoff_down) / (epsilon ** 2)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma
        original_sigma = process.sigma
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=process.N, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * process.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega


class CliquetOption(Option):
    """
    Cliquet (Ratchet) Option with multiple periods.
    """
    def __init__(self, K, m, T, is_call=True):
        """
        :param K: Initial strike price
        :param m: Number of sub-periods
        :param T: Total maturity
        :param is_call: True for Call, False for Put
        """
        self.K = K
        self.m = m
        self.T = T
        self.is_call = is_call

    def payoff(self, S):
        dt = self.T / self.m
        # Calculate returns for each sub-period
        returns = S[:, 1:] / S[:, :-1]
        # Calculate cumulative sum of excess returns
        if self.is_call:
            cum_excess = np.sum(np.maximum(returns - 1, 0), axis=1)
            return cum_excess
        else:
            cum_excess = np.sum(np.maximum(1 - returns, 0), axis=1)
            return cum_excess

    def greek_delta(self, S, epsilon=1e-4):
        # Finite difference approximation for Delta
        deltas = []
        for asset in range(S.shape[2]):
            S_up = S.copy()
            S_down = S.copy()
            S_up[:, 0, asset] += epsilon
            S_down[:, 0, asset] -= epsilon
            payoff_up = self.payoff(S_up)
            payoff_down = self.payoff(S_down)
            delta = (payoff_up - payoff_down) / (2 * epsilon)
            deltas.append(delta)
        return np.array(deltas)  # Shape: (d, M)

    def greek_gamma(self, S, epsilon=1e-4):
        # Finite difference approximation for Gamma
        deltas_up = self.greek_delta(S + epsilon, epsilon)
        deltas_down = self.greek_delta(S - epsilon, epsilon)
        gamma = (deltas_up - deltas_down) / (2 * epsilon)
        return gamma

    def greek_vega(self, process, option_price, d_sigma=1e-4):
        # Estimate Vega by bumping sigma for all assets
        original_sigma = process.sigma.copy()
        process.sigma += d_sigma
        S = process.generate_paths(S0=process.S0, T=process.T, N=self.m, M=process.M)
        payoff_up = self.payoff(S)
        price_up = np.mean(payoff_up) * np.exp(-process.r * self.T)
        vega = (price_up - option_price) / d_sigma
        process.sigma = original_sigma  # Reset sigma
        return vega
