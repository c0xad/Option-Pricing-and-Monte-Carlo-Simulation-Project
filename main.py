# main.py

import numpy as np
from stochastic_processes import GeometricBrownianMotion, HestonModel
from option_models import (
    EuropeanCallOption, EuropeanPutOption, AsianCallOption, BarrierOption,
    LookbackOption, AmericanOption, RainbowOption, SpreadOption, DigitalOption,
    CliquetOption
)
from monte_carlo_simulation import MonteCarloSimulator
from mlmc import MLMC_Simulator
from variance_reduction import AntitheticVariates, ControlVariates
from utils import black_scholes_price, plot_histogram
from option_models import EuropeanCallOption
from real_time_pricing import main as real_time_pricing_main
from real_time_pricing import fetch_market_data, calculate_historical_volatility, calculate_drift, price_options
import matplotlib.pyplot as plt
import os

def main():
    from real_time_pricing import fetch_market_data, calculate_historical_volatility, calculate_drift, price_options

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
        S0 = market_data['Close'].iloc[-1]  # Current stock price
        returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
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

        # Create a simulator instance for this ticker
        gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, r=r, seed=42)
        european_call = EuropeanCallOption(K=K)
        simulator = MonteCarloSimulator(process=gbm, option=european_call, T=T, N=N, M=M, r=r, seed=42)
        simulator.set_initial_price(S0)

        # Generate and save charts
        save_line_chart(simulator.get_price_path(), f'{ticker} - Price Path', f'{ticker}_price_path.png')
        save_line_chart(simulator.get_payoff_distribution(), f'{ticker} - Payoff Distribution', f'{ticker}_payoff_distribution.png')

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

    # Initialize stochastic process (Geometric Brownian Motion)
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, r=r, seed=42)

    # Initialize option (European Call)
    european_call = EuropeanCallOption(K=K)

    # Initialize Monte Carlo Simulator
    simulator = MonteCarloSimulator(process=gbm, option=european_call, T=T, N=N, M=M, r=r, seed=42)
    simulator.set_initial_price(S0)

    # Simulate without variance reduction
    price_mc, std_error_mc = simulator.simulate()
    print(f"European Call Option Price (MC): {price_mc:.4f} ± {1.96*std_error_mc:.4f} (95% CI)")

    # Analytical Black-Scholes price for comparison
    bs_price = black_scholes_price('call', S0, K, T, r, sigma)
    print(f"European Call Option Price (Black-Scholes): {bs_price:.4f}")

    # Simulate with Antithetic Variates
    antithetic = AntitheticVariates()
    price_av, std_error_av = simulator.simulate(method=antithetic)
    print(f"European Call Option Price (Antithetic Variates): {price_av:.4f} ± {1.96*std_error_av:.4f} (95% CI)")

    # Initialize Asian Call Option
    asian_call = AsianCallOption(K=K, averaging='arithmetic')
    simulator.option = asian_call
    price_asian, std_error_asian = simulator.simulate()
    print(f"Asian Call Option Price: {price_asian:.4f} ± {1.96*std_error_asian:.4f} (95% CI)")

    # Initialize Barrier Option
    barrier_option = BarrierOption(K=K, barrier=150, option_type='up-and-out', rebate=0.0)
    simulator.option = barrier_option
    price_barrier, std_error_barrier = simulator.simulate()
    print(f"Barrier Option Price: {price_barrier:.4f} ± {1.96*std_error_barrier:.4f} (95% CI)")

    # Initialize Lookback Option
    lookback_option = LookbackOption(option_type='fixed', K=100, is_call=True)
    simulator.option = lookback_option
    price_lookback, std_error_lookback = simulator.simulate()
    print(f"Fixed Strike Lookback Call Option Price: {price_lookback:.4f} ± {1.96*std_error_lookback:.4f} (95% CI)")

    # Initialize American Option
    american_option = AmericanOption(K=K, is_call=True)
    simulator.option = american_option
    price_american, std_error_american = simulator.simulate()
    print(f"American Call Option Price (LSM): {price_american:.4f} ± {1.96*std_error_american:.4f} (95% CI)")

    # Initialize Rainbow Option (Best-of)
    # For Rainbow Options, handle multiple assets
    S0_rainbow = [100, 100]
    gbm_rainbow = GeometricBrownianMotion(mu=mu, sigma=sigma, r=r, seed=42)
    
    rainbow_option = RainbowOption(K=100, weights=[0.5, 0.5], max_min='max')
    simulator = MonteCarloSimulator(process=gbm_rainbow, option=rainbow_option, T=T, N=N, M=M, r=r, seed=42)
    simulator.set_initial_price(S0_rainbow)
    price_rainbow, std_error_rainbow = simulator.simulate()
    print(f"Rainbow Option Price (Best-of): {price_rainbow:.4f} ± {1.96*std_error_rainbow:.4f} (95% CI)")

    # Initialize Spread Option
    spread_option = SpreadOption(K=0, asset_index_1=0, asset_index_2=1, is_call=True)
    simulator.option = spread_option
    price_spread, std_error_spread = simulator.simulate()
    print(f"Spread Option Price (Call on Spread): {price_spread:.4f} ± {1.96*std_error_spread:.4f} (95% CI)")

    # Initialize Digital Option
    digital_option = DigitalOption(K=100, payout=10.0, is_call=True)
    simulator.option = digital_option
    price_digital, std_error_digital = simulator.simulate()
    print(f"Digital Call Option Price: {price_digital:.4f} ± {1.96*std_error_digital:.4f} (95% CI)")

    # Initialize Cliquet Option
    cliquet_option = CliquetOption(K=100, m=12, T=1.0, is_call=True)
    simulator.option = cliquet_option
    price_cliquet, std_error_cliquet = simulator.simulate()
    print(f"Cliquet Option Price: {price_cliquet:.4f} ± {1.96*std_error_cliquet:.4f} (95% CI)")

    # Plot payoff distribution for European Call
    S = gbm.generate_paths(S0=S0, T=T, N=N, M=M, level=0)
    payoffs = european_call.payoff(S)
    discounted_payoffs = np.exp(-r*T) * payoffs
    plot_histogram(discounted_payoffs, title='European Call Option Payoff Distribution')

    # Implement MLMC
    print("\n--- Multi-Level Monte Carlo (MLMC) Simulation ---")
    # Define MLMC parameters
    M0 = 1000    # Initial samples per level
    L = 5        # Maximum level

    mlmc_simulator = MLMC_Simulator(
        process=gbm,
        option=european_call,
        T=T,
        N=N,
        M0=M0,
        L=L,
        r=r,
        seed=42
    )
    mlmc_simulator.set_initial_price(S0)
    price_mlmc, std_error_mlmc = mlmc_simulator.mlmc_estimator()
    print(f"European Call Option Price (MLMC): {price_mlmc:.4f} ± {1.96*std_error_mlmc:.4f} (95% CI)")

    # Generate and save charts for other calculations
    save_line_chart(simulator.get_price_path(), 'European Call - Price Path', 'european_call_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'European Call - Payoff Distribution', 'european_call_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Asian Call - Price Path', 'asian_call_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Asian Call - Payoff Distribution', 'asian_call_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Barrier Option - Price Path', 'barrier_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Barrier Option - Payoff Distribution', 'barrier_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Lookback Option - Price Path', 'lookback_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Lookback Option - Payoff Distribution', 'lookback_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'American Option - Price Path', 'american_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'American Option - Payoff Distribution', 'american_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Rainbow Option - Price Path', 'rainbow_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Rainbow Option - Payoff Distribution', 'rainbow_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Spread Option - Price Path', 'spread_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Spread Option - Payoff Distribution', 'spread_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Digital Option - Price Path', 'digital_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Digital Option - Payoff Distribution', 'digital_option_payoff_distribution.png')
    save_line_chart(simulator.get_price_path(), 'Cliquet Option - Price Path', 'cliquet_option_price_path.png')
    save_line_chart(simulator.get_payoff_distribution(), 'Cliquet Option - Payoff Distribution', 'cliquet_option_payoff_distribution.png')

    # Generate and save chart for MLMC
    save_line_chart(mlmc_simulator.get_mlmc_convergence(), 'MLMC Convergence', 'mlmc_convergence.png')

def save_line_chart(data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    
    # Create results folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig(os.path.join('results', filename))
    plt.close()

if __name__ == "__main__":
    main()
