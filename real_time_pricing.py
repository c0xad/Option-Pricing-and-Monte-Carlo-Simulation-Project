# real_time_pricing.py

import numpy as np
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from stochastic_processes import GeometricBrownianMotion, HestonModel
from option_models import EuropeanCallOption, AmericanOption
from monte_carlo_simulation import MonteCarloSimulator
from utils import black_scholes_price
import pandas as pd
import yfinance as yf

# Replace with your Alpha Vantage API key
API_KEY = "P824K3UZ2K5MXRVX"

def fetch_market_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    print(f"\nFirst few rows of data for {ticker}:")
    print(data.head())
    return data

def calculate_historical_volatility(returns):
    return np.std(returns) * np.sqrt(252)  # Annualized volatility

def calculate_drift(returns):
    return np.mean(returns) * 252  # Annualized drift

def price_options(S0, K, T, r, sigma, mu, N, M, ticker):
    # Initialize stochastic process (Geometric Brownian Motion)
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, r=r, seed=42)

    # Initialize options
    european_call = EuropeanCallOption(K=K)
    american_call = AmericanOption(K=K, is_call=True)

    # Initialize Monte Carlo Simulator
    simulator = MonteCarloSimulator(process=gbm, option=european_call, T=T, N=N, M=M, r=r, seed=42)
    simulator.set_initial_price(S0)

    # Price European Call
    price_mc, std_error_mc = simulator.simulate()
    
    # Price American Call
    simulator.option = american_call
    price_american, std_error_american = simulator.simulate()

    # Calculate Black-Scholes price for comparison
    bs_price = black_scholes_price('call', S0, K, T, r, sigma)

    print(f"\nPricing options for {ticker}:")
    print(f"Parameters: S0={S0:.2f}, K={K:.2f}, T={T:.4f}, r={r:.4f}, sigma={sigma:.4f}, mu={mu:.4f}")

    return price_mc, std_error_mc, price_american, std_error_american, bs_price

def main():
    # Parameters
    tickers = ["AAPL", "TSLA", "GOOGL"]  # Multiple stocks
    T = 30 / 252  # Time to maturity (30 trading days)
    N = 252  # Number of time steps
    M = 100000  # Number of simulations
    r = 0.05  # Risk-free rate (you may want to fetch this from a reliable source)

    for ticker in tickers:
        # Fetch market data
        market_data = fetch_market_data(ticker)
        print(f"\nMarket data for {ticker} fetched successfully.")

        # Calculate required parameters
        S0 = market_data['4. close'].iloc[0]  # Current stock price
        returns = np.log(market_data['4. close'] / market_data['4. close'].shift(1)).dropna()
        sigma = calculate_historical_volatility(returns)
        mu = calculate_drift(returns)

        # Set strike price based on current stock price
        K = round(S0, -1)  # Round to nearest 10

        print(f"\nCalculated parameters for {ticker}:")
        print(f"S0 = {S0:.2f}")
        print(f"sigma (volatility) = {sigma:.4f}")
        print(f"mu (drift) = {mu:.4f}")
        print(f"K (strike price) = {K:.2f}")

        # Price options
        price_mc, std_error_mc, price_american, std_error_american, bs_price = price_options(S0, K, T, r, sigma, mu, N, M, ticker)

        # Display results
        print(f"\nReal-time Option Pricing for {ticker}")
        print(f"Current Stock Price: ${S0:.2f}")
        print(f"Strike Price: ${K:.2f}")
        print(f"Historical Volatility: {sigma:.4f}")
        print(f"Drift: {mu:.4f}")
        print(f"European Call Option Price (MC): ${price_mc:.4f} ± ${1.96*std_error_mc:.4f} (95% CI)")
        print(f"American Call Option Price (LSM): ${price_american:.4f} ± ${1.96*std_error_american:.4f} (95% CI)")
        print(f"European Call Option Price (Black-Scholes): ${bs_price:.4f}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")