import numpy as np

class EnsembleRegressor:
    """
    Ensemble Regressor that combines predictions from multiple models.
    """
    def __init__(self, models, weights=None):
        """
        Initialize the ensemble with a list of trained models.
        
        Parameters:
            models (list): List of trained regression models.
            weights (list, optional): Weights for each model's prediction. If None, equal weights are used.
        """
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = weights
    
    def predict(self, X):
        """
        Predict using the ensemble of models.
        
        Parameters:
            X (np.ndarray): Input features.
        
        Returns:
            np.ndarray: Weighted average of predictions from all models.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        ensemble_prediction = np.dot(self.weights, predictions)
        return ensemble_prediction