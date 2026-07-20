import pytest
from sklearn.compose import ColumnTransformer

from src.data.generate_synthetic_data import generate_synthetic_clinical_requisitions
from src.features.build_features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    split_features_and_target,
    validate_feature_columns,
)


def test_split_features_and_target_excludes_leakage_columns() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=50, random_seed=201)

    X, y = split_features_and_target(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in X.columns
    assert not set(X.columns) & LEAKAGE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y) == 50


def test_validate_feature_columns_rejects_leakage_columns() -> None:
    with pytest.raises(ValueError, match="leakage"):
        validate_feature_columns(["test_type", "document_id"])

    with pytest.raises(ValueError, match="leakage"):
        validate_feature_columns(["ocr_confidence", TARGET_COLUMN])


def test_split_features_and_target_requires_target_column() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=20, random_seed=202)
    df = df.drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="requires_manual_review"):
        split_features_and_target(df)


def test_build_preprocessor_transforms_expected_features() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=30, random_seed=203)
    X, _ = split_features_and_target(df)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert isinstance(preprocessor, ColumnTransformer)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)
