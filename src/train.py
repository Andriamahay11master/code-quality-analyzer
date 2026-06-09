import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE


def load_featured_data():
    """Load the featured dataset from a CSV file."""
    input_path = os.path.join(PROCESSED_DATA_DIR, "featured_dataset.csv")
    return pd.read_csv(input_path)


def split_data(df, target_col="quality_label"):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def train_models(X_train, y_train):
    """Train multiple machine learning models and return the trained models."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and return a dictionary of evaluation metrics."""
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return results


def save_model(model, filename="best_model.pkl"):
    """Save the trained model to a file using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, model_path)
    return model_path


def choose_best_model(trained_models, X_test, y_test):
    """Evaluate all trained models and return the name, model, and results of the best performing model based on F1-score."""
    best_name = None
    best_model = None
    best_score = -1
    all_results = {}

    for name, model in trained_models.items():
        results = evaluate_model(model, X_test, y_test)
        all_results[name] = results

        if results["f1_score"] > best_score:
            best_score = results["f1_score"]
            best_name = name
            best_model = model

    return best_name, best_model, all_results


if __name__ == "__main__":
    """Run the training pipeline: load featured data, split, train models, evaluate, and save the best model."""
    df = load_featured_data()
    X_train, X_test, y_train, y_test = split_data(df)

    trained_models = train_models(X_train, y_train)
    best_name, best_model, all_results = choose_best_model(trained_models, X_test, y_test)
    model_path = save_model(best_model)

    print(f"Best model: {best_name}")
    print(f"Saved to: {model_path}")

    for name, results in all_results.items():
        print(f"\n=== {name} ===")
        print("Accuracy:", results["accuracy"])
        print("Precision:", results["precision"])
        print("Recall:", results["recall"])
        print("F1-score:", results["f1_score"])
        print(results["report"])
        print("Confusion Matrix:\n", results["confusion_matrix"])