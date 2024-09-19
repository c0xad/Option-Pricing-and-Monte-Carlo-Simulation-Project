# Advanced Option Pricing and Monte Carlo Simulation

## Overview

This project implements a comprehensive suite of option pricing models and Monte Carlo simulation techniques for financial derivatives. It includes real-time pricing capabilities, multiple option types, and advanced variance reduction methods. The code is designed for both educational purposes and practical applications in quantitative finance.

## Key Features

1. Multiple option types (European, American, Asian, Barrier, Lookback, Rainbow, Spread, Digital, Cliquet)
2. Stochastic processes (Geometric Brownian Motion, Heston Model, Jump-Diffusion)
3. Monte Carlo simulation with variance reduction techniques
4. Multi-Level Monte Carlo (MLMC) for improved efficiency
5. Real-time pricing using market data
6. Greeks calculation (Delta, Gamma, Vega)
7. Visualization of price paths and payoff distributions

## Why Use This Code?

1. **Comprehensive Coverage**: Implements a wide range of option types and pricing models, making it suitable for various financial instruments.
2. **Educational Tool**: Serves as a learning resource for students and professionals in quantitative finance.
3. **Performance Optimization**: Utilizes advanced techniques like MLMC and variance reduction for improved accuracy and efficiency.
4. **Real-world Application**: Integrates with real-time market data for practical use in trading and risk management.
5. **Flexibility**: Modular design allows for easy extension and customization.

## Key Formulas and Equations

1. **Black-Scholes Formula**:
   For European options:
   ```
   d1 = (ln(S/K) + (r + σ^2/2)T) / (σ√T)
   d2 = d1 - σ√T
   Call Price = S * N(d1) - K * e^(-rT) * N(d2)
   Put Price = K * e^(-rT) * N(-d2) - S * N(-d1)
   ```
   Where:
   - S: Current stock price
   - K: Strike price
   - r: Risk-free rate
   - σ: Volatility
   - T: Time to maturity
   - N(): Cumulative standard normal distribution function

2. **Geometric Brownian Motion**:
   ```
   dS = μSdt + σSdW
   ```
   Where:
   - μ: Drift
   - σ: Volatility
   - dW: Wiener process increment

3. **Monte Carlo Simulation**:
   Option Price ≈ e^(-rT) * E[max(S_T - K, 0)]
   Where E[] is the expected value over simulated paths.

4. **Variance Reduction Techniques**:
   - Antithetic Variates: E[f(X)] ≈ (f(X) + f(-X)) / 2
   - Control Variates: Y = X + c(Z - E[Z])

5. **Multi-Level Monte Carlo**:
   E[P_L] = E[P_0] + Σ(E[P_l - P_{l-1}]) for l = 1 to L

6. **Jump-Diffusion Process**:
   dS = (μ - λκ)Sdt + σSdW + JdN
   Where:
   - λ: Jump intensity
   - κ: Expected jump size
   - J: Random jump size
   - dN: Poisson process increment

## Usage
Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

Then
Run the main script to perform option pricing simulations: This will execute various option pricing scenarios and generate results and visualizations in the `results` folder.

