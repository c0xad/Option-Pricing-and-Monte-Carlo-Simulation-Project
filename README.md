# Option-Pricing-and-Monte-Carlo-Simulation-Project

Option Pricing and Monte Carlo Simulation Project
=================================================

This project implements various option pricing models and Monte Carlo simulation techniques for financial derivatives. It includes real-time pricing capabilities, multiple option types, and advanced variance reduction methods.

1. Overview
-----------
The project consists of several Python modules that work together to price options and simulate financial markets. The main components are:

- Main script (main.py)
- Real-time pricing module (real_time_pricing.py)
- Stochastic processes
- Option models
- Monte Carlo simulation
- Variance reduction techniques
- Utility functions

2. Option Types
---------------
The project supports various option types, including:

- European options (Call and Put)
- American options
- Asian options
- Barrier options
- Lookback options
- Rainbow options
- Spread options
- Digital options
- Cliquet options

3. Pricing Models
-----------------
3.1 Black-Scholes Model

The Black-Scholes model is used for European option pricing. The formula for a call option is:

C = S₀N(d₁) - Ke^(-rT)N(d₂)

Where:
C = Call option price
S₀ = Current stock price
K = Strike price
r = Risk-free rate
T = Time to maturity
N(x) = Cumulative standard normal distribution function
d₁ = (ln(S₀/K) + (r + σ²/2)T) / (σ√T)
d₂ = d₁ - σ√T

3.2 Monte Carlo Simulation

Monte Carlo simulation is used for pricing various option types. The general approach is:

1. Simulate asset price paths using a stochastic process (e.g., Geometric Brownian Motion)
2. Calculate option payoffs for each path
3. Average the discounted payoffs to estimate the option price

The Monte Carlo estimator for the option price is:

V̂ = e^(-rT) * (1/M) * Σ(i=1 to M) max(S_T^i - K, 0)

Where:
V̂ = Estimated option price
M = Number of simulations
S_T^i = Asset price at maturity for the i-th simulation

4. Stochastic Processes
-----------------------
4.1 Geometric Brownian Motion (GBM)

The project uses GBM to model asset price movements. The SDE for GBM is:

dS_t = μS_t dt + σS_t dW_t

Where:
S_t = Asset price at time t
μ = Drift
σ = Volatility
W_t = Wiener process

The discretized version for simulation is:

S_t = S_0 * exp((μ - σ²/2)t + σ√t * Z)

Where Z ~ N(0, 1) is a standard normal random variable.

5. Variance Reduction Techniques
--------------------------------
The project implements several variance reduction techniques to improve the efficiency of Monte Carlo simulations:

5.1 Antithetic Variates
5.2 Control Variates
5.3 Stratified Sampling

6. Multi-Level Monte Carlo (MLMC)
---------------------------------
MLMC is implemented to further improve the efficiency of option pricing. The MLMC estimator is:

Y = Σ(l=0 to L) (1/N_l) * Σ(i=1 to N_l) (P_l^i - P_{l-1}^i)

Where:
Y = MLMC estimator
L = Maximum level
N_l = Number of samples at level l
P_l^i = Payoff at level l for the i-th path

7. Real-time Pricing
--------------------
The project includes real-time pricing capabilities using market data fetched from Yahoo Finance. The process involves:

1. Fetching historical price data
2. Calculating historical volatility and drift
3. Pricing options using the calculated parameters

The real-time pricing functionality can be found in:
python:real_time_pricing.py
startLine: 16
endLine: 54

8. Usage
--------
To run the option pricing simulations, execute the main.py script:

python main.py

This will perform option pricing for multiple stocks and option types, displaying results and generating charts.

9. Dependencies
---------------
- NumPy
- Pandas
- Matplotlib
- yfinance

Install the required dependencies using:

pip install -r requirements.txt

10. Future Improvements
-----------------------
- Implement additional stochastic processes (e.g., Heston model)
- Add more exotic option types
- Enhance visualization capabilities
- Implement parallel computing for faster simulations
- Integrate with real-time market data APIs for live trading scenarios

