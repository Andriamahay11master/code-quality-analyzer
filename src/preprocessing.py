import os
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_data():
    """Load the dataset from CSV files."""
    repositories = pd.read_csv(os.path.join(RAW_DATA_DIR, "repositories.csv"))
    versions = pd.read_csv(os.path.join(RAW_DATA_DIR, "versions.csv"))
    attributes = pd.read_csv(os.path.join(RAW_DATA_DIR, "attribute-details.csv"))
    return repositories, versions, attributes


def basic_cleaning(df):
    """Perform basic cleaning steps on the DataFrame."""
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinite values if any
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def merge_datasets(repositories, versions, attributes):
    """Merge the repositories, versions, and attributes datasets into a single DataFrame."""
    repos = basic_cleaning(repositories)
    vers = basic_cleaning(versions)
    attrs = basic_cleaning(attributes)

    # Merge repositories with versions if a common key exists
    common_keys_rv = list(set(repos.columns).intersection(set(vers.columns)))
    if common_keys_rv:
        key = common_keys_rv[0]
        merged_df = pd.merge(repos, vers, on=key, how="inner")
    else:
        merged_df = repos.copy()

    # Merge with attributes if a common key exists
    common_keys_all = list(set(merged_df.columns).intersection(set(attrs.columns)))
    if common_keys_all:
        key = common_keys_all[0]
        merged_df = pd.merge(merged_df, attrs, on=key, how="inner")

    return merged_df


def handle_missing_values(df):
    """Handle missing values by filling numeric columns with median and categorical columns with mode."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def save_processed_data(df, filename="cleaned_dataset.csv"):
    """Save the processed DataFrame to a CSV file."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    return output_path


def preprocess_pipeline():
    """Run the full preprocessing pipeline: load data, merge, clean, and save."""
    repositories, versions, attributes = load_data()
    merged_df = merge_datasets(repositories, versions, attributes)
    cleaned_df = handle_missing_values(merged_df)
    output_path = save_processed_data(cleaned_df)
    return cleaned_df, output_path


if __name__ == "__main__":
    """Run the preprocessing pipeline and print the first few rows of the cleaned dataset."""
    cleaned_df, output_path = preprocess_pipeline()
    print(f"Processed dataset saved to: {output_path}")
    print(cleaned_df.head())