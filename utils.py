# utils.py

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

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
    
def interactive_plot_histogram(data, title='Payoff Distribution'):
    fig = make_subplots(rows=2, cols=1, subplot_titles=(title, 'Cumulative Distribution'))
    
    # Histogram
    fig.add_trace(go.Histogram(x=data, name='Payoff', nbinsx=50), row=1, col=1)
    
    # Cumulative Distribution
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(go.Scatter(x=sorted_data, y=cumulative, mode='lines', name='CDF'), row=2, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text='Discounted Payoffs', row=2, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Probability', row=2, col=1)
    
    fig.show()

def plot_option_surface(S_range, T_range, prices, title='Option Price Surface'):
    fig = go.Figure(data=[go.Surface(z=prices, x=S_range, y=T_range)])
    fig.update_layout(title=title, autosize=False,
                      width=800, height=600,
                      scene=dict(xaxis_title='Stock Price',
                                 yaxis_title='Time to Maturity',
                                 zaxis_title='Option Price'))
    fig.show()

def plot_correlation_heatmap(df, title='Feature Correlation Heatmap'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()
