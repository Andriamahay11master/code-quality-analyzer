import os
import joblib
import pandas as pd
from src.config import MODELS_DIR


def load_model(model_name="best_model.pkl"):
    """Load a trained model from a file using joblib."""
    model_path = os.path.join(MODELS_DIR, model_name)
    return joblib.load(model_path)


def load_new_data(file_path):
    """Load new data from a CSV file for prediction."""
    return pd.read_csv(file_path)


def predict_quality(model, new_data):
    """Use the loaded model to predict the quality label for new data."""
    predictions = model.predict(new_data)
    return predictions


if __name__ == "__main__":
    """Run the prediction pipeline: load the model, load new data, and make predictions."""
    model = load_model("best_model.pkl")

    # Example new data file path
    new_data_path = os.path.join("data", "processed", "sample_new_data.csv")
    new_data = load_new_data(new_data_path)

    predictions = predict_quality(model, new_data)

    print("Predictions:")
    print(predictions)