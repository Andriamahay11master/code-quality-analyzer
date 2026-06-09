import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import PROCESSED_DATA_DIR


def create_target_label(df):
    """Create a binary target label for code quality based on existing metrics or a fallback method."""
    df = df.copy()

    # If a target already exists, keep it
    if "quality_label" in df.columns:
        return df

    # Try to derive a target from common code quality-related columns
    possible_cols = [
        "cyclomatic_complexity",
        "CyclomaticComplexity",
        "loc",
        "LOC",
        "lines_of_code",
        "cbo",
        "CBO",
    ]

    existing_cols = [col for col in possible_cols if col in df.columns]

    if existing_cols:
        score = pd.Series(0, index=df.index, dtype=float)

        for col in existing_cols:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)

        threshold = score.median()
        df["quality_label"] = (score <= threshold).astype(int)
    else:
        # Fallback: create a dummy label from row order if no quality metric exists
        df["quality_label"] = (np.arange(len(df)) % 2).astype(int)

    return df


def encode_categorical_features(df):
    """Encode categorical features using Label Encoding, excluding the target label."""
    df = df.copy()
    label_encoders = {}

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        if col != "quality_label":
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders


def scale_numeric_features(df):
    """Scale numeric features using StandardScaler, excluding the target label."""
    df = df.copy()
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "quality_label" in numeric_cols:
        numeric_cols.remove("quality_label")

    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


def prepare_features(df):
    """Prepare the dataset for modeling by creating target labels, encoding categorical features, and scaling numeric features."""
    df = create_target_label(df)
    df, encoders = encode_categorical_features(df)
    df = df.dropna(subset=["quality_label"])
    df, scaler = scale_numeric_features(df)

    return df, encoders, scaler


def save_featured_data(df, filename="featured_dataset.csv"):
    """Save the featured DataFrame to a CSV file."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    """Run the feature engineering pipeline on the cleaned dataset and save the featured dataset."""
    input_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")
    df = pd.read_csv(input_path)

    featured_df, encoders, scaler = prepare_features(df)
    output_path = save_featured_data(featured_df)

    print(f"Featured dataset saved to: {output_path}")
    print(featured_df.head())