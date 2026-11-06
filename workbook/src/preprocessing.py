"""Data splitting and preprocessing helpers for the AI101 mobile-price project."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


TARGET_COLUMN = "price_range"
RANDOM_STATE = 42


def load_dataset(csv_path):
    """Load the original CSV and validate the expected target column."""
    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} was not found.")
    return data


def split_dataset(data, target_column=TARGET_COLUMN, random_state=RANDOM_STATE):
    """
    Create stratified 70:15:15 train/validation/test partitions.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    No imputer/scaler is fitted here; those must be fitted only on training data.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state + 1,
        stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_numeric_preprocessor():
    """
    Return a preprocessing pipeline for numeric predictors.
    Fit this object on X_train only. Median imputation is robust to skew;
    StandardScaler puts features on comparable scales for Logistic Regression.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def save_split_membership(X_train, X_val, X_test, output_path):
    """Save original row indices and their partition labels for reproducibility."""
    records = []
    for split_name, frame in (
        ("train", X_train), ("validation", X_val), ("test", X_test)
    ):
        records.extend(
            {"row_index": int(idx), "split": split_name} for idx in frame.index
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("row_index").to_csv(output_path, index=False)
