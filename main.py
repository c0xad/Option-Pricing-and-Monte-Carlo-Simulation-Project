# main.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from scipy.stats import norm
import mlflow
import mlflow.sklearn
import shap

# Import custom modules
from real_time_data import fetch_real_time_data
from fractional_nn import FractionalNeuralNetwork
from tda_features import compute_tda_features
from neural_sde import NeuralSDE, simulate_neural_sde, train_neural_sde

# Function to calculate cumulative distribution function for normal distribution
def norm_cdf(x):
    return norm.cdf(x)

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), save_path=None):
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve ({type(estimator).__name__})")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

# Function to simulate Geometric Brownian Motion
def simulate_gbm(S0, mu, sigma, T, N):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

# Function to generate dataset, integrating Neural SDE and Fractional Neural Network
def generate_dataset(tickers, T, N, M, r, simulations_per_ticker=100, use_neural_sde=False, neural_sde_model=None, use_fractional_nn=False, fractional_nn_model=None):
    X = []
    y = []
    price_series_list = []

    for ticker in tickers:
        print(f"\nFetching market data for {ticker}...")
        try:
            market_data = fetch_real_time_data(ticker)
        except ValueError as e:
            print(e)
            continue  # Skip this ticker if data fetching fails

        S0 = market_data['Close'].iloc[-1]
        K = round(S0, -1)

        if use_neural_sde and neural_sde_model is not None:
            print("Generating data using Neural SDE...")
            simulated_prices = simulate_neural_sde(neural_sde_model, S0, T, N)
            sigma = np.std(np.diff(np.log(simulated_prices)))
            mu = np.mean(np.diff(np.log(simulated_prices)))
        elif use_fractional_nn and fractional_nn_model is not None:
            print("Forecasting volatility using Fractional Neural Network...")
            # Prepare input features for the fractional NN
            features = np.array([S0, K, T, r]).reshape(1, -1)
            with torch.no_grad():
                sigma = fractional_nn_model(torch.tensor(features, dtype=torch.float32)).item()
            mu = r  # Assume risk-free rate as drift
        else:
            # Use historical data to calculate sigma and mu
            returns = np.log(market_data['Close'] / market_data['Close'].shift(1)).dropna()
            sigma = returns.std() * np.sqrt(252)  # Annualized volatility
            mu = returns.mean() * 252  # Annualized return

        print(f"\nCalculated parameters for {ticker}:")
        print(f"S0 = {S0:.2f}")
        print(f"sigma (volatility) = {sigma:.4f}")
        print(f"mu (drift) = {mu:.4f}")
        print(f"K (strike price) = {K:.2f}")

        print(f"\nPricing options for {ticker}:")
        print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, mu={mu}")

        for sim in range(simulations_per_ticker):
            # Simulate price path
            price_path = simulate_gbm(S0, mu, sigma, T, N)
            price_series_list.append(price_path)

            # Simulate option pricing using Black-Scholes formula
            d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            option_price = S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

            # Collect input features and target for surrogate models
            X.append([S0, K, T, r, sigma, mu])
            y.append(option_price)

        print(f"Collected {simulations_per_ticker} samples for {ticker}")

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal dataset size after generating: X: {X.shape}, y: {y.shape}")

    return X, y, price_series_list

# Function to compute TDA features (already implemented in tda_features.py)

# Function to augment dataset
def augment_dataset(X, y):
    """
    Augment the dataset by adding Gaussian noise to features and targets.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented feature matrix and target vector.
    """
    # Add Gaussian noise to features
    noise_X = np.random.normal(0, 0.01, X.shape)
    X_augmented = X + noise_X

    # Add Gaussian noise to targets
    noise_y = np.random.normal(0, 0.01, y.shape)
    y_augmented = y + noise_y

    return X_augmented, y_augmented

# Function to prepare data for WGAN
def prepare_wgan_data(X):
    """
    Prepare data loader for WGAN training.

    Parameters:
        X (np.ndarray): Feature matrix.

    Returns:
        DataLoader: PyTorch DataLoader for WGAN.
    """
    dataset = torch.utils.data.TensorDataset(torch.Tensor(X))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return data_loader

# WGAN components
class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=6):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def gradient_penalty(D, real_samples, fake_samples, device):
    """
    Calculate gradient penalty for WGAN-GP.

    Parameters:
        D (Discriminator): Discriminator model.
        real_samples (torch.Tensor): Real samples.
        fake_samples (torch.Tensor): Fake samples generated by Generator.
        device (str): Device to perform computations on.

    Returns:
        torch.Tensor: Gradient penalty.
    """
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan(data_loader, device='cpu', latent_dim=100, hidden_dim=128, lr=1e-4, lambda_gp=10, n_epochs=100, n_critic=5):
    """
    Train WGAN-GP for generating synthetic data.

    Parameters:
        data_loader (DataLoader): DataLoader containing real data.
        device (str): Device to train on.
        latent_dim (int): Dimension of latent space.
        hidden_dim (int): Hidden dimension for Generator and Discriminator.
        lr (float): Learning rate.
        lambda_gp (float): Gradient penalty coefficient.
        n_epochs (int): Number of training epochs.
        n_critic (int): Number of Discriminator updates per Generator update.

    Returns:
        Tuple[Generator, Discriminator]: Trained Generator and Discriminator models.
    """
    G = Generator(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=data_loader.dataset.tensors[0].shape[1]).to(device)
    D = Discriminator(input_dim=data_loader.dataset.tensors[0].shape[1], hidden_dim=hidden_dim).to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(n_epochs):
        for i, real_samples in enumerate(data_loader):
            real_samples = real_samples[0].to(device)
            batch_size = real_samples.size(0)

            # Train Discriminator
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_samples = G(z).detach()
                D_real = D(real_samples)
                D_fake = D(fake_samples)
                gp = gradient_penalty(D, real_samples.data, fake_samples.data, device=device)
                loss_D = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gp

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = G(z)
            loss_G = -torch.mean(D(fake_samples))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # Logging
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    return G, D

def generate_synthetic_data(generator, scaler, num_samples=10000, device='cpu'):
    """
    Generate synthetic data using the trained Generator.

    Parameters:
        generator (Generator): Trained Generator model.
        scaler (StandardScaler): Scaler fitted on real data.
        num_samples (int): Number of synthetic samples to generate.
        device (str): Device to perform computations on.

    Returns:
        np.ndarray: Synthetic data.
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.input_dim).to(device)
        synthetic_data = generator(z).cpu().numpy()
    synthetic_data = scaler.inverse_transform(synthetic_data)
    return synthetic_data

# Function to clean data
def clean_data(X, y):
    """
    Clean the dataset by removing NaNs and Infs.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned feature matrix and target vector.
    """
    print(f"\nInitial shapes: X: {X.shape}, y: {y.shape}")
    # Remove NaN values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"After removing NaNs: X: {X_clean.shape}, y: {y_clean.shape}")
    # Remove infinite values
    mask = ~np.isinf(X_clean).any(axis=1) & ~np.isinf(y_clean)
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]
    print(f"After removing infinities: X: {X_clean.shape}, y: {y_clean.shape}")
    return X_clean, y_clean

# Function to create pipeline
def create_pipeline(model):
    """
    Create a machine learning pipeline with scaling and the model.

    Parameters:
        model (Estimator): Scikit-learn estimator.

    Returns:
        Pipeline: Scikit-learn pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Function for Bayesian optimization
def bayesian_optimize(pipeline, param_space, X, y, n_iter=50, cv=5):
    """
    Perform Bayesian optimization for hyperparameter tuning.

    Parameters:
        pipeline (Pipeline): Scikit-learn pipeline.
        param_space (dict): Parameter space for BayesSearchCV.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        n_iter (int): Number of iterations.
        cv (int): Number of cross-validation folds.

    Returns:
        Tuple[Pipeline, float, dict]: Best pipeline, best score, best parameters.
    """
    bayes_cv = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=42
    )
    bayes_cv.fit(X, y)
    return bayes_cv.best_estimator_, bayes_cv.best_score_, bayes_cv.best_params_

# Function to cross-validate pipeline
def cross_validate_pipeline(pipeline, X, y, cv=5):
    """
    Perform cross-validation on the pipeline.

    Parameters:
        pipeline (Pipeline): Scikit-learn pipeline.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        cv (int): Number of cross-validation folds.

    Returns:
        float: Mean cross-validation score.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    return np.mean(scores)

# Function to save model
def save_model(model, filename):
    """
    Save the model to disk.

    Parameters:
        model (Estimator): Scikit-learn estimator.
        filename (str): Path to save the model.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Function to plot residuals and save plot
def plot_and_save_residuals(y_true, y_pred, model_name):
    """
    Plot residuals and save the plot.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        model_name (str): Name of the model.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title(f"{model_name} - Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plot_path = f"results/{model_name.replace(' ', '_').lower()}_residual_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to plot and save feature importance
def plot_and_save_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance and save the plot.

    Parameters:
        model (Estimator): Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.

    Returns:
        str: Path to the saved plot.
    """
    feature_importance = model.named_steps['model'].feature_importances_
    plt.figure(figsize=(12, 8))
    plt.bar(feature_names, feature_importance)
    plt.xticks(rotation=90)
    plt.title(f"Feature Importance ({model_name})")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plot_path = "results/random_forest_feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Add this new function for detailed asset-specific analysis
def analyze_asset(X, y, asset_name, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # Plotting
    plt.figure(figsize=(20, 15))
    
    # 1. SHAP summary plot
    plt.subplot(2, 2, 1)
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
    plt.title(f"SHAP Feature Importance for {asset_name}")
    
    # 2. Residual plot
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {asset_name}")
    
    # 3. Feature Importance plot
    plt.subplot(2, 2, 3)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance for {asset_name}")
    
    # 4. Actual vs Predicted plot
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {asset_name}")
    
    plt.tight_layout()
    plt.savefig(f"results/{asset_name}_analysis.png")
    plt.close()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rf_model, mse, r2, shap_values

# Main function
def main():
    # Start MLflow experiment
    mlflow.set_experiment("Option Pricing Models")
    with mlflow.start_run():
        # Parameters
        tickers = ["AAPL", "TSLA", "GOOGL"]  # Multiple stocks
        T = 30 / 252  # Time to maturity (30 trading days)
        N = 252  # Number of time steps
        M = 1000  # Number of simulations per option
        r = 0.05  # Risk-free rate
        simulations_per_ticker = 100  # Number of simulations per ticker

        # Create directories for models and results if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('datasets', exist_ok=True)

        # Train Neural SDE
        print("Training Neural SDE...")
        market_data_list = []
        for ticker in tickers:
            try:
                market_data = fetch_real_time_data(ticker)
                market_data_list.append(market_data)
            except ValueError as e:
                print(e)
                continue  # Skip this ticker if data fetching fails

        combined_prices = np.concatenate([md['Close'].values for md in market_data_list])
        times = np.arange(len(combined_prices)).reshape(-1,1)
        prices = combined_prices.reshape(-1,1)
        dataset = torch.utils.data.TensorDataset(torch.tensor(times, dtype=torch.float32),
                                                 torch.tensor(prices, dtype=torch.float32))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        neural_sde_model = NeuralSDE(input_dim=1, hidden_dim=16)
        train_neural_sde(neural_sde_model, data_loader, epochs=50, lr=1e-3)
        torch.save(neural_sde_model.state_dict(), 'models/neural_sde_model.pth')
        mlflow.log_param("NeuralSDE_epochs", 50)
        mlflow.log_param("NeuralSDE_learning_rate", 1e-3)
        mlflow.log_artifact("models/neural_sde_model.pth")

        # Train Fractional Neural Network
        print("\nTraining Fractional Neural Network...")
        X_frac = []
        y_frac = []
        for md in market_data_list:
            prices = md['Close'].values
            returns = np.diff(np.log(prices))
            volatilities = pd.Series(returns).rolling(window=30).std().dropna().values
            if len(volatilities) == 0:
                print("Insufficient data to compute volatility. Skipping Fractional Neural Network training for this ticker.")
                continue
            
            # Ensure all arrays have the same length
            n = min(len(prices) - 60, len(volatilities))
            
            features = np.array([
                prices[30:30+n],  # prices[30:-30]
                prices[60:60+n],  # prices[60:]
                np.full(n, T),
                np.full(n, r)
            ]).T
            
            X_frac.append(features)
            y_frac.append(volatilities[:n])

        if X_frac and y_frac:
            X_frac = np.vstack(X_frac)
            y_frac = np.concatenate(y_frac)
            X_frac_tensor = torch.tensor(X_frac, dtype=torch.float32)
            y_frac_tensor = torch.tensor(y_frac, dtype=torch.float32).unsqueeze(1)
            
            fractional_nn_model = FractionalNeuralNetwork(input_dim=4, hidden_dims=[64, 64], output_dim=1, alpha=0.5)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fractional_nn_model.parameters(), lr=1e-3)
            
            epochs = 50
            batch_size = 32
            dataset_frac = torch.utils.data.TensorDataset(X_frac_tensor, y_frac_tensor)
            data_loader_frac = torch.utils.data.DataLoader(dataset_frac, batch_size=batch_size, shuffle=True)
            
            fractional_nn_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_batch in data_loader_frac:
                    optimizer.zero_grad()
                    outputs = fractional_nn_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch+1) % 10 == 0 or epoch == 0:
                    avg_loss = total_loss / len(data_loader_frac)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            torch.save(fractional_nn_model.state_dict(), 'models/fractional_nn_model.pth')
            mlflow.log_param("FractionalNN_epochs", epochs)
            mlflow.log_param("FractionalNN_learning_rate", 1e-3)
            mlflow.log_artifact("models/fractional_nn_model.pth")
        else:
            fractional_nn_model = None
            print("No sufficient data to train Fractional Neural Network.")

        # Generate dataset
        print("\nGenerating new dataset...")
        X, y, price_series_list = generate_dataset(
            tickers, T, N, M, r, simulations_per_ticker=simulations_per_ticker,
            use_neural_sde=True, neural_sde_model=neural_sde_model,
            use_fractional_nn=True, fractional_nn_model=fractional_nn_model
        )

        # Compute TDA features
        print("\nComputing TDA features...")
        tda_features = compute_tda_features(price_series_list)  # Pass price_series_list
        if tda_features.shape[0] != X.shape[0]:
            print("Mismatch between TDA features and samples. Adjusting...")
            min_samples = min(tda_features.shape[0], X.shape[0])
            X = X[:min_samples]
            y = y[:min_samples]
            tda_features = tda_features[:min_samples]
        X = np.hstack((X, tda_features))

        # Save dataset
        data = np.column_stack((X, y))
        df = pd.DataFrame(data)
        dataset_path = 'datasets/simulation_data.csv'
        df.to_csv(dataset_path, index=False)
        print(f"Dataset saved to {dataset_path}")
        mlflow.log_artifact(dataset_path)

        # Data augmentation
        print("\nAugmenting dataset...")
        X_augmented, y_augmented = augment_dataset(X, y)

        # WGAN training
        print("\nTraining WGAN...")
        scaler = StandardScaler()
        X_train_wgan = scaler.fit_transform(X_augmented)
        wgan_data_loader = prepare_wgan_data(X_train_wgan)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        G, D = train_wgan(wgan_data_loader, device=device, n_epochs=50)
        torch.save(G.state_dict(), 'models/wgan_generator.pth')
        torch.save(D.state_dict(), 'models/wgan_discriminator.pth')
        mlflow.log_param("WGAN_epochs", 50)
        mlflow.log_param("WGAN_latent_dim", 100)
        mlflow.log_artifact("models/wgan_generator.pth")
        mlflow.log_artifact("models/wgan_discriminator.pth")

        # Generate synthetic data
        print("\nGenerating synthetic data using WGAN...")
        synthetic_X = generate_synthetic_data(G, scaler, num_samples=1000, device=device)
        print(f"Synthetic data shape: {synthetic_X.shape}")

        # Combine original and synthetic data
        X_combined = np.vstack((X_augmented, synthetic_X))
        y_combined = np.hstack((y_augmented, np.random.choice(y_augmented, size=synthetic_X.shape[0])))
        print(f"Combined dataset size: X: {X_combined.shape}, y: {y_combined.shape}")

        # Clean data
        print("\nCleaning combined dataset...")
        X_clean, y_clean = clean_data(X_combined, y_combined)

        # Split data
        print("\nSplitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        print(f"Training set size: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Testing set size: X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Define parameter spaces
        gp_param_space = {
            'model__alpha': Real(1e-10, 1e-1, prior='log-uniform'),
            'model__n_restarts_optimizer': Integer(0, 10)
        }
        rf_param_space = {
            'model__n_estimators': Integer(50, 200),
            'model__max_depth': Integer(3, 15),
            'model__min_samples_split': Integer(2, 10),
            'model__min_samples_leaf': Integer(1, 10)
        }
        mlp_param_space = {
            'model__hidden_layer_sizes': Categorical([(50,), (100,), (150,), (200,)]),
            'model__alpha': Real(1e-5, 1e-1, prior='log-uniform'),
            'model__learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform')
        }

        # Create pipelines
        gp_pipeline = create_pipeline(GaussianProcessRegressor())
        rf_pipeline = create_pipeline(RandomForestRegressor(random_state=42))
        mlp_pipeline = create_pipeline(MLPRegressor(random_state=42))

        # Optimize models
        for name, pipeline, param_space in [
            ("Gaussian Process", gp_pipeline, gp_param_space),
            ("Random Forest", rf_pipeline, rf_param_space),
            ("MLP", mlp_pipeline, mlp_param_space)
        ]:
            print(f"\nOptimizing {name}...")
            try:
                best_model, score, params = bayesian_optimize(
                    pipeline, param_space, X_train, y_train, n_iter=10, cv=5
                )
                if best_model is not None:
                    print(f"Best {name} score: {score:.4f}")
                    print(f"Best {name} parameters: {params}")
                    mlflow.log_params(params)
                    mlflow.log_metric(f"{name} CV Score", score)
                    cv_score = cross_validate_pipeline(best_model, X_train, y_train, cv=5)
                    print(f"{name} Cross-validation score: {cv_score:.4f}")
                    mlflow.log_metric(f"{name} Cross-validation Score", cv_score)
                    model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
                    save_model(best_model, model_filename)
                    # Log model with input example
                    input_example = X_train[:1]
                    mlflow.sklearn.log_model(best_model, f"{name}_model", input_example=input_example)
                else:
                    print(f"{name} optimization failed.")
            except Exception as e:
                print(f"An unexpected error occurred during {name} optimization: {e}")

        # Ensemble Modeling
        print("\nTraining Ensemble Model...")
        try:
            gp_model = joblib.load('models/gaussian_process_model.pkl')
            rf_model = joblib.load('models/random_forest_model.pkl')
            mlp_model = joblib.load('models/mlp_model.pkl')
            estimators = [
                ('gp', gp_model),
                ('rf', rf_model),
                ('mlp', mlp_model)
            ]
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=RandomForestRegressor(random_state=42)
            )
            ensemble.fit(X_train, y_train)
            ensemble_score = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
            print(f"Ensemble Model Cross-validation Score: {ensemble_score:.4f}")
            mlflow.log_metric("Ensemble Model Cross-validation Score", ensemble_score)
            save_model(ensemble, 'models/ensemble_model.pkl')
            # Log ensemble model with input example
            mlflow.sklearn.log_model(ensemble, "ensemble_model", input_example=X_train[:1])
        except Exception as e:
            print(f"Failed to train or log Ensemble Model: {e}")

        # Evaluate models on the test set
        print("\nEvaluating models on the test set...")
        for name in ["Gaussian Process", "Random Forest", "MLP", "Ensemble"]:
            model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    print(f"{name} Test MSE: {mse:.4f}")
                    mlflow.log_metric(f"{name} Test MSE", mse)

                    # Plot residuals and save
                    residual_plot_path = plot_and_save_residuals(y_test, predictions, name)
                    mlflow.log_artifact(residual_plot_path)

                    if name == "Random Forest":
                        # Plot feature importance
                        feature_names = ['S0', 'K', 'T', 'r', 'sigma', 'mu'] + [f"tda_{i}" for i in range(X.shape[1] - 6)]
                        feature_importance_plot = plot_and_save_feature_importance(model, feature_names, name)
                        mlflow.log_artifact(feature_importance_plot)

                        # Explainable AI with SHAP
                        explainer = shap.TreeExplainer(model.named_steps['model'])
                        shap_values = explainer.shap_values(X_test)
                        shap_summary_path = "results/random_forest_shap_summary.png"
                        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
                        plt.savefig(shap_summary_path)
                        plt.close()
                        mlflow.log_artifact(shap_summary_path)
                except Exception as e:
                    print(f"Failed to load or evaluate {name} model: {e}")
            else:
                print(f"{name} model file not found. Skipping evaluation.")

        # After generating the dataset
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            X_ticker = X[X['ticker'] == ticker].drop('ticker', axis=1)
            y_ticker = y[X['ticker'] == ticker]
            
            feature_names = X_ticker.columns.tolist()
            
            with mlflow.start_run(run_name=f"{ticker}_analysis"):
                model, mse, r2, shap_values = analyze_asset(X_ticker, y_ticker, ticker, feature_names)
                
                # Log metrics
                mlflow.log_metric(f"{ticker}_MSE", mse)
                mlflow.log_metric(f"{ticker}_R2", r2)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{ticker}_model")
                
                # Log plots
                mlflow.log_artifact(f"results/{ticker}_analysis.png")
                
                # Log feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                feature_importance.to_csv(f"results/{ticker}_feature_importance.csv", index=False)
                mlflow.log_artifact(f"results/{ticker}_feature_importance.csv")
                
                # Log SHAP values
                shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
                shap_values_df.to_csv(f"results/{ticker}_shap_values.csv", index=False)
                mlflow.log_artifact(f"results/{ticker}_shap_values.csv")

        print("\nAll processes completed.")
        mlflow.end_run()

if __name__ == "__main__":
    main()