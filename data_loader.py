import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(filename='datasets/simulation_data.csv', test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespaces
        print("Available columns in the dataset:", df.columns.tolist())
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{filename}' does not exist. Please provide a valid file path.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{filename}' is empty. Please provide a valid dataset.")
    except pd.errors.ParserError:
        raise ValueError(f"The file '{filename}' could not be parsed. Please check the CSV format.")

    columns = ['S0', 'K', 'T', 'r', 'sigma', 'mu']
    if 'garch_volatility' in df.columns:
        columns.append('garch_volatility')
    
    target_column = 'price_mc'  # Define the target column name

    # Check if the target column exists
    if target_column not in df.columns:
        print("Error: Target column not found.")
        print("Available columns:", df.columns.tolist())
        raise KeyError(f"'{target_column}' column not found in the dataset. Please check the column names.")
    
    X = df[columns].values
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X, y