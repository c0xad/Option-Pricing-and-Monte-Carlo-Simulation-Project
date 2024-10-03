# neural_sde.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralSDE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralSDE, self).__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positivity
        )

    def forward(self, t, x):
        mu = self.mu_net(x)
        sigma = self.sigma_net(x)
        return mu, sigma

def simulate_neural_sde(model, S0, T, N):
    """
    Simulate asset prices using the trained Neural SDE.

    Parameters:
        model (NeuralSDE): Trained Neural SDE model.
        S0 (float): Initial asset price.
        T (float): Time horizon.
        N (int): Number of time steps.

    Returns:
        np.ndarray: Simulated asset prices.
    """
    dt = T / N
    times = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = S0
    model.eval()
    with torch.no_grad():
        for i in range(1, N):
            t = torch.tensor([times[i-1]], dtype=torch.float32)
            x = torch.tensor([S[i-1]], dtype=torch.float32)
            mu, sigma = model(t, x)
            dW = np.random.normal(0, np.sqrt(dt))
            S[i] = S[i-1] + mu.item() * S[i-1] * dt + sigma.item() * S[i-1] * dW
    return S

def train_neural_sde(model, data_loader, epochs=100, lr=1e-3):
    """
    Train the Neural SDE model.

    Parameters:
        model (NeuralSDE): Neural SDE model to train.
        data_loader (DataLoader): DataLoader containing training data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for times, prices in data_loader:
            optimizer.zero_grad()
            mu_pred, sigma_pred = model(times, prices)
            # Assume zero drift and unit volatility for simplicity
            mu_true = torch.zeros_like(mu_pred)
            sigma_true = torch.ones_like(sigma_pred)
            loss = loss_fn(mu_pred, mu_true) + loss_fn(sigma_pred, sigma_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
