from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor

def kernel_from_string(kernel_str):
    if kernel_str == 'RBF':
        return RBF()
    elif kernel_str == 'Matern':
        return Matern()
    else:
        raise ValueError(f"Unknown kernel: {kernel_str}")

class GPRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='RBF', alpha=1e-10, n_restarts_optimizer=0, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        kernel_obj = kernel_from_string(self.kernel)
        self.model = GaussianProcessRegressor(
            kernel=kernel_obj,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'kernel': self.kernel,
            'alpha': self.alpha,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def bayesian_optimize(pipeline, param_space, X, y, n_iter=50, cv=3):
    if len(X) < cv:
        print(f"Not enough samples for cross-validation. Samples: {len(X)}, CV folds: {cv}")
        return None, None, None

    try:
        optimizer = BayesSearchCV(
            pipeline,
            param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            error_score='raise'
        )
        optimizer.fit(X, y)
        return optimizer.best_estimator_, optimizer.best_score_, optimizer.best_params_
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Pipeline: {pipeline}")
        print(f"Parameter space: {param_space}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
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