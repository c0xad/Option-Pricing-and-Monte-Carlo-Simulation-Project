import numpy as np
from scipy.stats import norm

def generate_synthetic_data(X, y, n_synthetic=1000):
    n_features = X.shape[1]
    synthetic_X = np.zeros((n_synthetic, n_features))
    synthetic_y = np.zeros(n_synthetic)

    for i in range(n_synthetic):
        # Randomly select two parent samples
        idx1, idx2 = np.random.choice(X.shape[0], 2, replace=False)
        
        # Interpolate between parents
        alpha = np.random.random()
        synthetic_X[i] = alpha * X[idx1] + (1 - alpha) * X[idx2]
        
        # Add some noise to features
        noise = np.random.normal(0, 0.05, n_features)
        synthetic_X[i] += noise
        
        # Interpolate target and add noise
        synthetic_y[i] = alpha * y[idx1] + (1 - alpha) * y[idx2]
        synthetic_y[i] += np.random.normal(0, 0.05 * np.std(y))

    return synthetic_X, synthetic_y

def augment_dataset(X, y, augmentation_factor=2):
    if len(y) <= 1:
        print("Warning: Not enough data points for augmentation. Returning original dataset.")
        return X, y
    
    n_synthetic = int(X.shape[0] * (augmentation_factor - 1))
    synthetic_X, synthetic_y = generate_synthetic_data(X, y, n_synthetic)
    
    augmented_X = np.vstack((X, synthetic_X))
    augmented_y = np.concatenate((y, synthetic_y))
    
    print(f"Augmented dataset shapes: X: {augmented_X.shape}, y: {augmented_y.shape}")
    
    return augmented_X, augmented_y