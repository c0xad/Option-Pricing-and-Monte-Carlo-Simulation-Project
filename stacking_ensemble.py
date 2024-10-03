from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from surrogate_models import (
    GaussianProcessRegressor, RandomForestRegressor, MLPRegressor,
    SVR, GradientBoostingRegressor, Ridge
)

def create_stacking_ensemble(base_models, final_estimator=LinearRegression()):
    """
    Create a stacking ensemble regressor.

    Parameters:
        base_models (list): List of (name, estimator) tuples.
        final_estimator: The estimator to use as the final estimator.

    Returns:
        stacking_regressor: The stacking ensemble model.
    """
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        n_jobs=-1
    )
    return stacking_regressor

def train_stacking_ensemble(X_train, y_train):
    """
    Train a stacking ensemble with optimized base models.

    Parameters:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.

    Returns:
        stacking_model: Trained stacking ensemble model.
    """
    base_models = [
        ('gp', GaussianProcessRegressor()),
        ('rf', RandomForestRegressor()),
        ('mlp', MLPRegressor()),
        ('svr', SVR()),
        ('gb', GradientBoostingRegressor()),
        ('ridge', Ridge())
    ]

    stacking_model = create_stacking_ensemble(base_models)
    stacking_model.fit(X_train, y_train)
    return stacking_model