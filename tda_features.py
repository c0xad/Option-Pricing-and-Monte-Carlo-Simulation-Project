# tda_features.py

import numpy as np
import gudhi as gd
from gudhi.representations import Landscape

def compute_tda_features(price_series_list):
    """
    Compute topological features from price series using persistent homology.

    Parameters:
        price_series_list (list of np.ndarray): List of price series arrays for each sample.

    Returns:
        np.ndarray: Topological features for each sample.
    """
    tda_features = []
    landscape = Landscape(num_landscapes=5, resolution=50)

    for idx, prices in enumerate(price_series_list):
        try:
            # Ensure prices is a 1D array
            prices = prices.flatten()
            # Normalize prices
            if np.std(prices) == 0:
                print(f"Sample {idx}: Standard deviation is zero. Appending zero features.")
                tda_features.append(np.zeros(250))  # 5 landscapes * 50 resolution
                continue
            prices = (prices - np.mean(prices)) / np.std(prices)
            # Use sliding window to create point cloud
            window_size = 10  # Adjust window size as needed
            if len(prices) < window_size:
                # Not enough data for TDA
                print(f"Sample {idx}: Not enough data for TDA. Appending zero features.")
                tda_features.append(np.zeros(250))
                continue
            point_cloud = np.array([prices[i:i + window_size] for i in range(len(prices) - window_size + 1)])
            # Build Rips complex
            rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=2.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            # Compute persistence diagram
            simplex_tree.compute_persistence()
            persistence_pairs = simplex_tree.persistence_intervals_in_dimension(1)
            if len(persistence_pairs) == 0:
                print(f"Sample {idx}: Empty persistence diagram. Appending zero features.")
                tda_features.append(np.zeros(250))
                continue
            # Extract features using persistence landscapes
            landscape_features = landscape.fit_transform([persistence_pairs])
            tda_features.append(landscape_features[0].flatten())
        except Exception as e:
            print(f"Error computing TDA features for sample {idx}: {e}")
            tda_features.append(np.zeros(250))  # Fallback to zeros

    tda_features = np.array(tda_features)
    return tda_features
