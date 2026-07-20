"""Validate the synthetic clinical requisition dataset.

This module checks that the generated CSV follows the expected schema and that
derived fields still match the documented synthetic validation rules.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data import (
    BOOLEAN_COLUMNS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RECORD_COUNT,
    EXPECTED_COLUMNS,
    SPECIMEN_TYPES,
    TEST_TYPES,
    apply_manual_review_rules,
    calculate_missing_field_count,
)


CLASS_IMBALANCE_WARNING_THRESHOLD = 0.20


@dataclass(frozen=True)
class ValidationResult:
    """Summary of dataset validation checks."""

    path: Path | None
    record_count: int
    column_count: int
    missing_value_count: int
    duplicate_document_id_count: int
    manual_review_rate: float
    class_counts: dict[bool, int]
    warnings: list[str]
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        """Return True when validation found no errors."""
        return not self.errors


def load_dataset(csv_path: str | Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load a synthetic requisition CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file cannot be parsed as a usable CSV.
    """
    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    try:
        return pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {dataset_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset file could not be parsed as CSV: {dataset_path}") from exc
    except OSError as exc:
        raise ValueError(f"Dataset file could not be read: {dataset_path}") from exc


def _append_missing_column_errors(
    df: pd.DataFrame,
    required_columns: list[str],
    errors: list[str],
) -> None:
    """Add readable errors for required columns missing from a DataFrame."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")


def validate_dataset(
    df: pd.DataFrame,
    expected_count: int | None = DEFAULT_RECORD_COUNT,
    path: str | Path | None = None,
) -> ValidationResult:
    """Validate schema, quality checks, and synthetic rule-derived columns.

    Args:
        df: Dataset to validate.
        expected_count: Expected number of records. Use None to skip this check.
        path: Optional source path included in the returned result.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    source_path = Path(path) if path is not None else None

    record_count = len(df)
    column_count = len(df.columns)
    missing_value_count = int(df.isna().sum().sum())
    duplicate_document_id_count = (
        int(df["document_id"].duplicated().sum()) if "document_id" in df.columns else 0
    )
    manual_review_rate = 0.0
    class_counts: dict[bool, int] = {False: 0, True: 0}

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append("Dataset columns do not match the expected schema or order.")
        _append_missing_column_errors(df, EXPECTED_COLUMNS, errors)

    if expected_count is not None and record_count != expected_count:
        errors.append(f"Expected {expected_count} records, found {record_count}.")

    if missing_value_count > 0:
        errors.append(f"Dataset contains {missing_value_count} missing values.")

    if "document_id" in df.columns:
        if duplicate_document_id_count > 0:
            errors.append(
                f"document_id contains {duplicate_document_id_count} duplicate values."
            )
        if not df["document_id"].astype(str).str.match(r"^REQ-[A-Z0-9]{4}-[0-9]{8}$").all():
            errors.append("document_id values do not match the expected synthetic format.")

    if "test_type" in df.columns:
        unsupported_test_types = sorted(set(df["test_type"].dropna()) - set(TEST_TYPES))
        if unsupported_test_types:
            errors.append(f"Unsupported test_type values: {', '.join(unsupported_test_types)}")

    if "specimen_type" in df.columns:
        unsupported_specimen_types = sorted(
            set(df["specimen_type"].dropna()) - set(SPECIMEN_TYPES)
        )
        if unsupported_specimen_types:
            errors.append(
                f"Unsupported specimen_type values: {', '.join(unsupported_specimen_types)}"
            )

    for column in BOOLEAN_COLUMNS:
        if column in df.columns and not pd.api.types.is_bool_dtype(df[column]):
            errors.append(f"{column} must be a boolean column.")

    if "ocr_confidence" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["ocr_confidence"]):
            errors.append("ocr_confidence must be numeric.")
        elif not df["ocr_confidence"].between(0.0, 1.0, inclusive="both").all():
            errors.append("ocr_confidence must be between 0.0 and 1.0.")

    if "missing_field_count" in df.columns and all(
        column in df.columns for column in EXPECTED_COLUMNS
    ):
        expected_missing_field_count = calculate_missing_field_count(df)
        if not df["missing_field_count"].equals(expected_missing_field_count):
            errors.append(
                "missing_field_count does not match required field presence columns."
            )

    if "requires_manual_review" in df.columns:
        value_counts = df["requires_manual_review"].value_counts().to_dict()
        class_counts = {
            False: int(value_counts.get(False, 0)),
            True: int(value_counts.get(True, 0)),
        }
        if record_count > 0 and pd.api.types.is_bool_dtype(df["requires_manual_review"]):
            manual_review_rate = float(df["requires_manual_review"].mean())

            minority_rate = min(manual_review_rate, 1.0 - manual_review_rate)
            if minority_rate < CLASS_IMBALANCE_WARNING_THRESHOLD:
                warnings.append(
                    "Possible class imbalance: one class represents less than "
                    f"{CLASS_IMBALANCE_WARNING_THRESHOLD:.0%} of records."
                )

    review_rule_columns = [
        "date_of_birth_present",
        "provider_present",
        "collection_date_present",
        "diagnosis_code_present",
        "duplicate_kit",
        "ocr_confidence",
        "requires_manual_review",
    ]
    if all(column in df.columns for column in review_rule_columns):
        expected_review_labels = apply_manual_review_rules(df)
        if not df["requires_manual_review"].equals(expected_review_labels):
            errors.append(
                "requires_manual_review does not match the documented synthetic rules."
            )

    return ValidationResult(
        path=source_path,
        record_count=record_count,
        column_count=column_count,
        missing_value_count=missing_value_count,
        duplicate_document_id_count=duplicate_document_id_count,
        manual_review_rate=manual_review_rate,
        class_counts=class_counts,
        warnings=warnings,
        errors=errors,
    )


def format_validation_result(result: ValidationResult) -> str:
    """Format a validation result for command-line output."""
    source = str(result.path) if result.path is not None else "<in-memory DataFrame>"
    status = "PASSED" if result.is_valid else "FAILED"
    lines = [
        f"Validation status: {status}",
        f"Dataset: {source}",
        f"Rows: {result.record_count}",
        f"Columns: {result.column_count}",
        f"Missing values: {result.missing_value_count}",
        f"Duplicate document IDs: {result.duplicate_document_id_count}",
        f"Manual-review rate: {result.manual_review_rate:.2%}",
        f"Class counts: {result.class_counts}",
    ]

    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)

    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate the synthetic clinical requisition dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV path to validate. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=DEFAULT_RECORD_COUNT,
        help=f"Expected row count. Default: {DEFAULT_RECORD_COUNT}.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset validation from the command line."""
    args = parse_args()

    try:
        df = load_dataset(args.input)
        result = validate_dataset(
            df,
            expected_count=args.expected_records,
            path=args.input,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print(format_validation_result(result))
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
