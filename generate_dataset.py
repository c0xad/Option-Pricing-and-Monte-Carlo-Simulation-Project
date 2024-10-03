# Example data generation script

import numpy as np
import pandas as pd

def monte_carlo_simulation(S0, K, T, r, sigma, mu, num_simulations=1000):
    """
    Perform Monte Carlo simulation to estimate option price.
    
    Parameters:
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        mu (float): Drift.
        num_simulations (int): Number of simulation runs.
    
    Returns:
        float: Estimated option price.
    """
    dt = T
    # Simulate end stock price
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(num_simulations))
    # Calculate option payoff
    payoffs = np.maximum(S_T - K, 0)  # For a Call option
    # Discount payoffs back to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

def generate_dataset(num_samples=10000, filename='datasets/simulation_data.csv'):
    np.random.seed(42)
    data = {
        'S0': np.random.uniform(50, 150, num_samples),
        'K': np.random.uniform(50, 150, num_samples),
        'T': np.random.uniform(0.1, 2, num_samples),  # Time to maturity in years
        'r': np.random.uniform(0.01, 0.05, num_samples),  # Risk-free rate
        'sigma': np.random.uniform(0.1, 0.5, num_samples),  # Volatility
        'mu': np.random.uniform(0.05, 0.15, num_samples)  # Drift
    }

    df = pd.DataFrame(data)
    
    # Calculate 'price_mc' using Monte Carlo simulation
    df['price_mc'] = df.apply(lambda row: monte_carlo_simulation(
        row['S0'], row['K'], row['T'], row['r'], row['sigma'], row['mu']
    ), axis=1)
    
    # Optionally, add GARCH volatility if needed
    # df['garch_volatility'] = calculate_garch_volatility(...)

    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    generate_dataset()