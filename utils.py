# utils.py

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os

def black_scholes_price(option_type, S0, K, T, r, sigma):
    """
    Computes the Black-Scholes price for European options.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return price

def plot_histogram(data, title, filename=None):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    if filename:
        plt.savefig(os.path.join('results', filename))
        plt.close()
    else:
        plt.show()
