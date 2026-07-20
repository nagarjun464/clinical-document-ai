"""Generate synthetic clinical requisition records for manual-review modeling.

The generated data is intentionally synthetic. It contains no patient names,
real dates of birth, addresses, medical record numbers, scanned documents,
clinical notes, employer documents, or proprietary business rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_clinical_requisitions.csv"
DEFAULT_RECORD_COUNT = 1_000
DEFAULT_RANDOM_SEED = 42

OCR_CONFIDENCE_REVIEW_THRESHOLD = 0.82
MISSING_FIELD_REVIEW_THRESHOLD = 2

TEST_TYPES = (
    "Complete Blood Count",
    "Comprehensive Metabolic Panel",
    "Lipid Panel",
    "Thyroid Panel",
    "Genetic Screening",
    "Infectious Disease Panel",
)

SPECIMEN_TYPES = (
    "Blood",
    "Saliva",
    "Urine",
    "Buccal Swab",
    "Tissue",
)

REQUIRED_FIELD_COLUMNS = (
    "date_of_birth_present",
    "provider_present",
    "collection_date_present",
    "diagnosis_code_present",
)

BOOLEAN_COLUMNS = REQUIRED_FIELD_COLUMNS + (
    "duplicate_kit",
    "requires_manual_review",
)

EXPECTED_COLUMNS = [
    "document_id",
    "test_type",
    "specimen_type",
    "date_of_birth_present",
    "provider_present",
    "collection_date_present",
    "diagnosis_code_present",
    "duplicate_kit",
    "ocr_confidence",
    "missing_field_count",
    "requires_manual_review",
]


def calculate_missing_field_count(df: pd.DataFrame) -> pd.Series:
    """Count missing required fields for each synthetic requisition."""
    missing_columns = [column for column in REQUIRED_FIELD_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Cannot calculate missing fields. Missing columns: {missing_text}")

    return df.loc[:, REQUIRED_FIELD_COLUMNS].eq(False).sum(axis=1).astype(int)


def apply_manual_review_rules(df: pd.DataFrame) -> pd.Series:
    """Apply documented synthetic validation rules to create the review label."""
    required_columns = set(REQUIRED_FIELD_COLUMNS) | {"duplicate_kit", "ocr_confidence"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Cannot apply manual-review rules. Missing columns: {missing_text}")

    missing_field_count = calculate_missing_field_count(df)

    return (
        df["duplicate_kit"].eq(True)
        | df["ocr_confidence"].lt(OCR_CONFIDENCE_REVIEW_THRESHOLD)
        | missing_field_count.ge(MISSING_FIELD_REVIEW_THRESHOLD)
    )


def validate_record_count(record_count: int) -> None:
    """Validate the requested number of synthetic records."""
    if not isinstance(record_count, int):
        raise TypeError("record_count must be an integer.")
    if record_count <= 0:
        raise ValueError("record_count must be greater than zero.")


def validate_dataframe(df: pd.DataFrame, expected_count: int) -> None:
    """Validate schema, ranges, and rule-derived fields in the dataset."""
    validate_record_count(expected_count)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match the expected schema.")

    if len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} records, found {len(df)}.")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df["document_id"].is_unique:
        raise ValueError("document_id values must be unique.")

    if not df["document_id"].str.match(r"^REQ-[A-Z0-9]{4}-[0-9]{8}$").all():
        raise ValueError("document_id values do not match the expected synthetic format.")

    if not df["test_type"].isin(TEST_TYPES).all():
        raise ValueError("Dataset contains an unsupported test_type value.")

    if not df["specimen_type"].isin(SPECIMEN_TYPES).all():
        raise ValueError("Dataset contains an unsupported specimen_type value.")

    for column in BOOLEAN_COLUMNS:
        if not pd.api.types.is_bool_dtype(df[column]):
            raise TypeError(f"{column} must be a boolean column.")

    if not df["ocr_confidence"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("ocr_confidence must be between 0.0 and 1.0.")

    expected_missing_field_count = calculate_missing_field_count(df)
    if not df["missing_field_count"].equals(expected_missing_field_count):
        raise ValueError("missing_field_count does not match required field presence columns.")

    expected_review_labels = apply_manual_review_rules(df)
    if not df["requires_manual_review"].equals(expected_review_labels):
        raise ValueError("requires_manual_review does not match the documented synthetic rules.")


def build_fake_document_ids(record_count: int, random_seed: int) -> list[str]:
    """Build reproducible, unique synthetic document IDs."""
    Faker.seed(random_seed)
    fake = Faker("en_US")
    fake.seed_instance(random_seed)
    fake.unique.clear()

    return [
        fake.unique.bothify(text="REQ-????-########").upper()
        for _ in range(record_count)
    ]


def generate_synthetic_clinical_requisitions(
    record_count: int = DEFAULT_RECORD_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a validated synthetic clinical requisition dataset.

    Args:
        record_count: Number of synthetic records to create.
        random_seed: Random seed used for reproducible data generation.

    Returns:
        A pandas DataFrame containing synthetic requisition records.

    Raises:
        TypeError: If record_count is not an integer.
        ValueError: If record_count is less than one or validation fails.
    """
    validate_record_count(record_count)

    rng = np.random.default_rng(random_seed)

    df = pd.DataFrame(
        {
            "document_id": build_fake_document_ids(record_count, random_seed),
            "test_type": rng.choice(
                TEST_TYPES,
                size=record_count,
                p=[0.22, 0.22, 0.16, 0.14, 0.12, 0.14],
            ),
            "specimen_type": rng.choice(
                SPECIMEN_TYPES,
                size=record_count,
                p=[0.46, 0.18, 0.18, 0.12, 0.06],
            ),
            "date_of_birth_present": rng.random(record_count) < 0.97,
            "provider_present": rng.random(record_count) < 0.93,
            "collection_date_present": rng.random(record_count) < 0.90,
            "diagnosis_code_present": rng.random(record_count) < 0.88,
            "duplicate_kit": rng.random(record_count) < 0.04,
            "ocr_confidence": np.round(0.55 + (rng.beta(8, 2, size=record_count) * 0.45), 3),
        }
    )

    df["missing_field_count"] = calculate_missing_field_count(df)
    df["requires_manual_review"] = apply_manual_review_rules(df)
    df = df.loc[:, EXPECTED_COLUMNS]

    validate_dataframe(df, expected_count=record_count)
    return df


def save_synthetic_data(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    record_count: int = DEFAULT_RECORD_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    """Generate and save synthetic requisition data to a CSV file."""
    output_file = Path(output_path)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_clinical_requisitions(
            record_count=record_count,
            random_seed=random_seed,
        )
        df.to_csv(output_file, index=False)
    except OSError as exc:
        raise OSError(f"Unable to write synthetic data to {output_file}: {exc}") from exc

    return output_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data-generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical requisition data for manual-review modeling."
    )
    parser.add_argument(
        "--records",
        type=int,
        default=DEFAULT_RECORD_COUNT,
        help=f"Number of synthetic records to generate. Default: {DEFAULT_RECORD_COUNT}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible output. Default: {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the synthetic data-generation workflow."""
    args = parse_args()

    try:
        output_file = save_synthetic_data(
            output_path=args.output,
            record_count=args.records,
            random_seed=args.seed,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Data generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {args.records} synthetic records at {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
