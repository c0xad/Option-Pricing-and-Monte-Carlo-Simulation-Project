import gpytorch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from ensemble_models import EnsembleRegressor
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import Optimizer as BayesianOptimization
from sklearn.model_selection import learning_curve
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add engineered features
        moneyness = X[:, 0] / X[:, 1]  # S0 / K
        time_to_maturity = X[:, 2]
        implied_volatility = X[:, 4] * np.sqrt(time_to_maturity)
        
        return np.column_stack((X, moneyness, implied_volatility))

def create_pipeline(model):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return pipeline

def train_pipeline(model, X_train, y_train):
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)

def cross_validate_pipeline(pipeline, X, y, cv=5):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error')
    return np.sqrt(-scores.mean())

def train_gp_model(X, y):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10)
    gp.fit(X, y)
    return gp

def train_random_forest(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

def train_mlp(X, y):
    """
    Train a complex MLP Regressor with enhanced architecture and training parameters.

    Parameters:
        X (np.ndarray): Training feature data.
        y (np.ndarray): Training target data.

    Returns:
        MLPRegressor: Trained MLP regressor.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(200, 200, 100),      # More hidden layers and neurons
        activation='tanh',                       # Activation function
        solver='adam',                           # Optimizer
        alpha=0.001,                             # L2 regularization parameter
        learning_rate='adaptive',                # Adaptive learning rate
        learning_rate_init=0.001,                # Initial learning rate
        max_iter=1000,                            # Maximum iterations
        early_stopping=True,                     # Enable early stopping
        validation_fraction=0.1,                 # Fraction for validation in early stopping
        n_iter_no_change=50,                     # Number of iterations with no improvement to wait before stopping
        random_state=42,                         # Reproducibility
        verbose=True                             # Enable verbose output
    )
    mlp.fit(X, y)
    return mlp

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

# Add Bayesian Neural Network
class BayesianNeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim=100, epochs=100, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        return model
    
    def fit(self, X, y):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(torch.tensor(X, dtype=torch.float32)).squeeze()
            loss = self.criterion(outputs, torch.tensor(y, dtype=torch.float32))
            loss.backward()
            self.optimizer.step()
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32)).squeeze()
        return outputs.numpy()

# Add Deep Gaussian Process
class DeepGaussianProcess(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DeepGaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_deep_gp_model(X, y, iterations=100):
    import gpytorch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_x = torch.tensor(X, dtype=torch.float32).to(device)
    train_y = torch.tensor(y, dtype=torch.float32).to(device)
    
    likelihood = GaussianLikelihood().to(device)
    model = DeepGaussianProcess(train_x, train_y, likelihood).to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.3f}')
    
    model.eval()
    likelihood.eval()
    return model, likelihood

def predict_deep_gp(model, likelihood, X_new):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.tensor(X_new, dtype=torch.float32)
        preds = likelihood(model(test_x))
        return preds.mean.numpy()

# Function to train Bayesian Neural Network
def train_bnn_model(X, y, hidden_dim=100, epochs=100, lr=0.001):
    bnn = BayesianNeuralNetwork(input_dim=X.shape[1], hidden_dim=hidden_dim, epochs=epochs, lr=lr)
    bnn.fit(X, y)
    return bnn

# Function to train Deep Gaussian Process
def train_deep_gp(X, y, iterations=100):
    model, likelihood = train_deep_gp_model(X, y, iterations=iterations)
    return model, likelihood

# Function to evaluate Deep Gaussian Process
def evaluate_deep_gp(model, likelihood, X_test):
    preds = predict_deep_gp(model, likelihood, X_test)
    return preds

# Function to train all models including advanced ones
def train_all_models_extended(X_train, y_train):
    models = {}
    
    # Existing models
    models['gp'] = train_gp_model(X_train, y_train)
    models['rf'] = train_random_forest(X_train, y_train)
    models['mlp'] = train_mlp(X_train, y_train)
    models['svr'] = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1).fit(X_train, y_train)
    models['gb'] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(X_train, y_train)
    models['ridge'] = Ridge(alpha=1.0).fit(X_train, y_train)
    
    # Advanced models
    models['bnn'] = train_bnn_model(X_train, y_train)
    models['deep_gp'], models['deep_gp_likelihood'] = train_deep_gp(X_train, y_train)
    
    return models

# Function to evaluate all models including advanced ones
def evaluate_all_models_extended(models, X_test, y_test, scaler=None):
    rmse_scores = {}
    
    for name, model in models.items():
        if name in ['gp', 'mlp', 'bnn', 'deep_gp']:
            if scaler:
                if name == 'deep_gp':
                    preds = evaluate_deep_gp(model, models['deep_gp_likelihood'], X_test)
                else:
                    X_test_scaled = scaler.transform(X_test)
                    if name == 'bnn':
                        preds = model.predict(X_test_scaled)
                    else:
                        preds = model.predict(X_test_scaled)
            else:
                preds = model.predict(X_test)
        else:
            preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        rmse_scores[name] = rmse
    
    return rmse_scores

# Function to create and save ensemble model
def train_and_save_ensemble(models, weights, filename='models/ensemble_model.joblib'):
    ensemble = EnsembleRegressor(models=models, weights=weights)
    save_model(ensemble, filename)
    return ensemble

# Function to load and use ensemble model
def load_and_use_ensemble(filename):
    return load_model(filename)

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, search_type='grid', n_iter=50):
    """
    Tune hyperparameters using Grid Search or Randomized Search.

    Parameters:
        model: The machine learning model to tune.
        param_grid (dict): Dictionary with parameter names as keys and lists of parameter settings to try as values.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        cv (int): Number of cross-validation folds.
        search_type (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
        n_iter (int): Number of parameter settings sampled for RandomizedSearchCV.

    Returns:
        best_estimator: The best model found by the search.
        best_params: The best parameters found by the search.
    """
    if search_type == 'grid':
        search_cv = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
    elif search_type == 'random':
        search_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    search_cv.fit(X_train, y_train)
    print(f"Best parameters: {search_cv.best_params_}")
    print(f"Best score: {search_cv.best_score_:.4f}")

    return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_

def optimize_models_parallel(X_train, y_train):
    """
    Optimize hyperparameters for all models in parallel.

    Parameters:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.

    Returns:
        optimized_models (dict): Dictionary of optimized models.
    """
    optimized_models = {}

    # Define model and parameter grids
    models_params = {
        'gp': {
            'model': GaussianProcessRegressor(),
            'param_grid': {
                'kernel': [C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2))],
                'n_restarts_optimizer': [10, 15, 20],
                'alpha': [1e-10, 1e-8, 1e-6]
            },
            'search_type': 'grid'
        },
        'rf': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'search_type': 'random',
            'n_iter': 10
        },
        'mlp': {
            'model': MLPRegressor(
                hidden_layer_sizes=(200, 200, 100),
                activation='tanh',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=42,
                verbose=True
            ),
            'param_grid': {
                'hidden_layer_sizes': [(50, 50), (200, 200, 100), (300, 300, 200)],
                'activation': ['tanh', 'relu'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'search_type': 'random',
            'n_iter': 20
        },
        'svr': {
            'model': SVR(),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1, 10],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'search_type': 'random',
            'n_iter': 15
        },
        'gb': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'search_type': 'grid'
        },
        'ridge': {
            'model': Ridge(random_state=42),
            'param_grid': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'search_type': 'grid'
        }
    }

    def tune_and_store(name, details):
        best_model, best_score, best_params = tune_hyperparameters(
            model=details['model'],
            param_grid=details['param_grid'],
            X_train=X_train,
            y_train=y_train,
            cv=5,
            search_type=details.get('search_type', 'grid'),
            n_iter=details.get('n_iter', 50)  # Default n_iter for random search
        )
        optimized_models[name] = best_model
        print(f"Optimized {name} Params: {best_params}")

    # Optimize all models in parallel
    Parallel(n_jobs=-1)(
        delayed(tune_and_store)(name, details) for name, details in models_params.items()
    )

    return optimized_models

class GPRegressorWrapper(GaussianProcessRegressor):
    def __init__(self, random_state=None, n_restarts_optimizer=0, max_iter=1000):
        super().__init__(
            kernel=None,
            alpha=1e-10,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            normalize_y=True,
            copy_X_train=True,
            n_targets=None
        )
        self.max_iter = max_iter

def bayesian_optimize(model, param_space, X_train, y_train, n_iter=50, cv=5):
    """
    Perform Bayesian optimization for hyperparameter tuning.

    Parameters:
        model: The machine learning model to optimize.
        param_space (dict): Dictionary with parameter names as keys and ranges as values.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        n_iter (int): Number of iterations for optimization.
        cv (int): Number of cross-validation folds.

    Returns:
        best_model: The best model found.
        best_score (float): The best score achieved.
        best_params (dict): The best parameters found.
    """
    try:
        optimizer = BayesSearchCV(
            model,
            param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            error_score='raise'
        )
        optimizer.fit(X_train, y_train)
        return optimizer.best_estimator_, optimizer.best_score_, optimizer.best_params_
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Model: {model}")
        print(f"Parameter space: {param_space}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return None, None, None

# Define parameter spaces for different models
gp_param_space = {
    'model__kernel': Categorical(['RBF', 'Matern']),
    'model__alpha': Real(1e-10, 1e+1, prior='log-uniform'),
    'model__n_restarts_optimizer': Integer(0, 10)
}

rf_param_space = {
    'model__n_estimators': Integer(10, 1000),
    'model__max_depth': Integer(1, 30),
    'model__min_samples_split': Integer(2, 30),
    'model__min_samples_leaf': Integer(1, 30)
}

mlp_param_space = {
    'model__hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
    'model__activation': Categorical(['relu', 'tanh']),
    'model__alpha': Real(1e-5, 1e-2, prior='log-uniform'),
    'model__learning_rate': Categorical(['constant', 'adaptive']),
    'model__max_iter': Integer(100, 500)
}