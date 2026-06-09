import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR


def load_data():
    """Load the featured dataset from a CSV file."""
    input_path = os.path.join(PROCESSED_DATA_DIR, "featured_dataset.csv")
    return pd.read_csv(input_path)


def load_model(model_name="best_model.pkl"):
    """Load a trained model from a file using joblib."""
    model_path = os.path.join(MODELS_DIR, model_name)
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and return a dictionary of evaluation metrics."""
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return results


def save_metrics(results, filename="metrics.csv"):
    """Save the evaluation metrics to a CSV file."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    output_path = os.path.join(METRICS_DIR, filename)

    metrics_df = pd.DataFrame([
        {
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "report": results["report"],
            "confusion_matrix": str(results["confusion_matrix"]),
        }
    ])

    metrics_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    """Run the evaluation pipeline: load the model, load the featured dataset, evaluate the model, and save the metrics."""
    df = load_data()
    model = load_model()

    target_col = "quality_label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    results = evaluate_model(model, X, y)
    metrics_path = save_metrics(results)

    print(f"Metrics saved to: {metrics_path}")
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1-score:", results["f1_score"])
    print(results["report"])