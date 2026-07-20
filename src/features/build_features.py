"""Build model features for synthetic requisition review prediction."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "requires_manual_review"

LEAKAGE_COLUMNS = {
    "document_id",
    "patient_name",
    TARGET_COLUMN,
}

CATEGORICAL_FEATURES = [
    "test_type",
    "specimen_type",
]

BOOLEAN_FEATURES = [
    "date_of_birth_present",
    "provider_present",
    "collection_date_present",
    "diagnosis_code_present",
    "duplicate_kit",
]

NUMERIC_FEATURES = [
    "ocr_confidence",
    "missing_field_count",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + BOOLEAN_FEATURES + NUMERIC_FEATURES


def cast_boolean_features_to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean feature columns to numeric values for scikit-learn."""
    return X.astype(float)


def validate_feature_columns(feature_columns: Sequence[str]) -> None:
    """Validate that feature columns do not include obvious leakage columns."""
    leakage_columns = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leakage_columns:
        leakage_text = ", ".join(leakage_columns)
        raise ValueError(f"Feature columns contain leakage columns: {leakage_text}")


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset has the columns needed for model training."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required modeling columns: {missing_text}")


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into safe features and the target label."""
    validate_required_columns(df)
    validate_feature_columns(FEATURE_COLUMNS)

    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    if not pd.api.types.is_bool_dtype(y):
        raise TypeError(f"{TARGET_COLUMN} must be a boolean target column.")

    return X, y


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build a preprocessing transformer for categorical, boolean, and numeric inputs."""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    boolean_pipeline = Pipeline(
        steps=[
            ("cast_to_float", FunctionTransformer(cast_boolean_features_to_float)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("boolean", boolean_pipeline, BOOLEAN_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
