from pathlib import Path

import pandas as pd
import pytest

from src.data.generate_synthetic_data import (
    DEFAULT_RECORD_COUNT,
    EXPECTED_COLUMNS,
    MISSING_FIELD_REVIEW_THRESHOLD,
    OCR_CONFIDENCE_REVIEW_THRESHOLD,
    apply_manual_review_rules,
    calculate_missing_field_count,
    generate_synthetic_clinical_requisitions,
    save_synthetic_data,
    validate_dataframe,
)


def test_default_dataset_has_expected_shape_and_columns() -> None:
    df = generate_synthetic_clinical_requisitions()

    assert len(df) == DEFAULT_RECORD_COUNT
    assert list(df.columns) == EXPECTED_COLUMNS


def test_document_ids_are_unique_and_synthetic() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=100, random_seed=7)

    assert df["document_id"].is_unique
    assert df["document_id"].str.match(r"^REQ-[A-Z0-9]{4}-[0-9]{8}$").all()


def test_missing_field_count_is_derived_from_presence_columns() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=100, random_seed=11)

    expected_missing_count = calculate_missing_field_count(df)

    pd.testing.assert_series_equal(
        df["missing_field_count"],
        expected_missing_count,
        check_names=False,
    )


def test_manual_review_rules_are_applied() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=200, random_seed=13)

    expected_review_label = (
        df["duplicate_kit"]
        | (df["ocr_confidence"] < OCR_CONFIDENCE_REVIEW_THRESHOLD)
        | (df["missing_field_count"] >= MISSING_FIELD_REVIEW_THRESHOLD)
    )

    pd.testing.assert_series_equal(
        df["requires_manual_review"],
        expected_review_label,
        check_names=False,
    )


def test_manual_review_rule_boundaries() -> None:
    df = pd.DataFrame(
        {
            "date_of_birth_present": [True, True, True, False],
            "provider_present": [True, True, True, False],
            "collection_date_present": [True, True, True, True],
            "diagnosis_code_present": [True, True, True, True],
            "duplicate_kit": [False, True, False, False],
            "ocr_confidence": [0.82, 0.99, 0.81, 0.99],
        }
    )

    labels = apply_manual_review_rules(df)

    assert labels.tolist() == [False, True, True, True]


def test_generation_is_reproducible_with_same_seed() -> None:
    first = generate_synthetic_clinical_requisitions(record_count=50, random_seed=123)
    second = generate_synthetic_clinical_requisitions(record_count=50, random_seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_invalid_record_count_raises_error() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        generate_synthetic_clinical_requisitions(record_count=0)


def test_validation_rejects_tampered_missing_field_count() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=25, random_seed=21)
    df.loc[0, "missing_field_count"] = 99

    with pytest.raises(ValueError, match="missing_field_count"):
        validate_dataframe(df, expected_count=25)


def test_save_synthetic_data_creates_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic.csv"

    saved_path = save_synthetic_data(
        output_path=output_path,
        record_count=20,
        random_seed=5,
    )

    assert saved_path == output_path
    assert output_path.exists()

    saved_df = pd.read_csv(output_path)
    assert len(saved_df) == 20
    assert list(saved_df.columns) == EXPECTED_COLUMNS
