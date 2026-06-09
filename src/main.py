from src.preprocessing import preprocess_pipeline
from src.feature_engineering import prepare_features, save_featured_data
from src.train import load_featured_data, split_data, train_models, choose_best_model, save_model
from src.evaluate import evaluate_model, save_metrics


def main():
    """Run the main pipeline: preprocessing, feature engineering, training, and evaluation."""
    print("Starting preprocessing...")
    cleaned_df, cleaned_path = preprocess_pipeline()
    print(f"Cleaned data saved to: {cleaned_path}")

    print("Starting feature engineering...")
    featured_df, encoders, scaler = prepare_features(cleaned_df)
    featured_path = save_featured_data(featured_df)
    print(f"Featured data saved to: {featured_path}")

    print("Starting training...")
    X_train, X_test, y_train, y_test = split_data(featured_df)
    trained_models = train_models(X_train, y_train)
    best_name, best_model, all_results = choose_best_model(trained_models, X_test, y_test)
    model_path = save_model(best_model)
    print(f"Best model: {best_name}")
    print(f"Model saved to: {model_path}")

    print("Evaluating best model...")
    results = evaluate_model(best_model, X_test, y_test)
    metrics_path = save_metrics(results)
    print(f"Metrics saved to: {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    """Run the main pipeline: preprocessing, feature engineering, training, and evaluation."""
    main()