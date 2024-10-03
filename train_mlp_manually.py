import numpy as np
from surrogate_models import train_mlp, save_model
from data_loader import load_dataset

def main():
    try:
        # Load the dataset
        X_train, X_test, y_train, y_test, scaler, X_all, y_all = load_dataset()
    except KeyError as e:
        print(f"Data loading error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return

    # Train the enhanced MLP model
    print("Training Enhanced MLP Model...")
    try:
        mlp_model = train_mlp(X_train, y_train)
    except Exception as e:
        print(f"An error occurred during MLP training: {e}")
        return
    print("Training completed.")

    # Save the trained MLP model
    model_filename = "models/mlp_model.pkl"
    try:
        save_model(mlp_model, model_filename)
        print(f"MLP model saved to {model_filename}")
    except Exception as e:
        print(f"An error occurred while saving the MLP model: {e}")

if __name__ == "__main__":
    main()