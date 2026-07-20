from pathlib import Path

import pandas as pd
import pytest

from src.data.generate_synthetic_data import (
    EXPECTED_COLUMNS,
    generate_synthetic_clinical_requisitions,
)
from src.data.validate_dataset import (
    format_validation_result,
    load_dataset,
    validate_dataset,
)


def test_load_dataset_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    df = generate_synthetic_clinical_requisitions(record_count=10, random_seed=100)
    df.to_csv(csv_path, index=False)

    loaded_df = load_dataset(csv_path)

    assert len(loaded_df) == 10
    assert list(loaded_df.columns) == EXPECTED_COLUMNS


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_dataset(missing_path)


def test_validate_dataset_accepts_valid_generated_data() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=100, random_seed=101)

    result = validate_dataset(df, expected_count=100)

    assert result.is_valid
    assert result.record_count == 100
    assert result.column_count == len(EXPECTED_COLUMNS)
    assert 0.0 <= result.manual_review_rate <= 1.0
    assert result.errors == []


def test_validate_dataset_reports_missing_column() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=25, random_seed=102)
    df = df.drop(columns=["ocr_confidence"])

    result = validate_dataset(df, expected_count=25)

    assert not result.is_valid
    assert any("Missing required columns: ocr_confidence" in error for error in result.errors)


def test_validate_dataset_detects_duplicate_document_ids() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=25, random_seed=103)
    df.loc[1, "document_id"] = df.loc[0, "document_id"]

    result = validate_dataset(df, expected_count=25)

    assert not result.is_valid
    assert any("duplicate" in error.lower() for error in result.errors)


def test_validate_dataset_detects_tampered_missing_field_count() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=25, random_seed=104)
    df.loc[0, "missing_field_count"] = 99

    result = validate_dataset(df, expected_count=25)

    assert not result.is_valid
    assert any("missing_field_count" in error for error in result.errors)


def test_validate_dataset_detects_tampered_manual_review_label() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=25, random_seed=105)
    df.loc[0, "requires_manual_review"] = not df.loc[0, "requires_manual_review"]

    result = validate_dataset(df, expected_count=25)

    assert not result.is_valid
    assert any("requires_manual_review" in error for error in result.errors)


def test_validate_dataset_warns_about_possible_class_imbalance() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=50, random_seed=106)
    df["date_of_birth_present"] = True
    df["provider_present"] = True
    df["collection_date_present"] = True
    df["diagnosis_code_present"] = True
    df["duplicate_kit"] = False
    df["ocr_confidence"] = 0.95
    df["missing_field_count"] = 0
    df["requires_manual_review"] = False

    result = validate_dataset(df, expected_count=50)

    assert result.is_valid
    assert result.manual_review_rate == 0.0
    assert result.class_counts == {False: 50, True: 0}
    assert any("class imbalance" in warning.lower() for warning in result.warnings)


def test_format_validation_result_includes_status_and_counts() -> None:
    df = generate_synthetic_clinical_requisitions(record_count=10, random_seed=107)
    result = validate_dataset(df, expected_count=10, path="synthetic.csv")

    summary = format_validation_result(result)

    assert "Validation status: PASSED" in summary
    assert "Rows: 10" in summary
    assert "Manual-review rate:" in summary
