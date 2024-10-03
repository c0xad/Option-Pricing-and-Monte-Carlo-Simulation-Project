# econometric_models.py

import numpy as np
from arch import arch_model
import pandas as pd
import plotly.graph_objects as go

class GARCHModel:
    def __init__(self, returns, p=1, q=1):
        self.returns = returns
        self.p = p
        self.q = q
        self.model = None
        self.results = None

    def fit(self):
        self.model = arch_model(self.returns, vol='Garch', p=self.p, q=self.q)
        self.results = self.model.fit(disp='off')
        return self.results

    def forecast(self, horizon=10):
        if self.results is None:
            raise ValueError("Model must be fit before forecasting")
        return self.results.forecast(horizon=horizon)

    def plot_volatility(self):
        if self.results is None:
            raise ValueError("Model must be fit before plotting")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.returns.index, y=self.returns,
                                 mode='lines', name='Returns'))
        fig.add_trace(go.Scatter(x=self.returns.index, y=np.sqrt(self.results.conditional_volatility),
                                 mode='lines', name='GARCH Volatility'))
        fig.update_layout(title='Returns and GARCH Volatility',
                          xaxis_title='Date',
                          yaxis_title='Returns / Volatility')
        fig.show()

def calculate_garch_volatility(returns, p=1, q=1):
    garch = GARCHModel(returns, p=p, q=q)
    results = garch.fit()
    forecast = garch.forecast(horizon=1)
    return np.sqrt(forecast.variance.values[-1, :])[0]