import os
import pandas as pd
from src.config import RAW_DATA_DIR

def load_dataset():
    """Load the Software Code Quality and Source Code Metrics Dataset from CSV files."""
    repositories_path = os.path.join(RAW_DATA_DIR, "repositories.csv")
    versions_path = os.path.join(RAW_DATA_DIR, "versions.csv")
    attributes_path = os.path.join(RAW_DATA_DIR, "attribute-details.csv")

    repositories = pd.read_csv(repositories_path)
    versions = pd.read_csv(versions_path)
    attributes = pd.read_csv(attributes_path)

    return repositories, versions, attributes


def quick_overview(df, name="DataFrame"):
    """Print a quick overview of the DataFrame including shape, head, missing values, and columns."""
    print(f"\n{name} shape: {df.shape}")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    """Run a quick overview of the loaded datasets."""
    repositories, versions, attributes = load_dataset()

    quick_overview(repositories, "Repositories")
    quick_overview(versions, "Versions")
    quick_overview(attributes, "Attributes")