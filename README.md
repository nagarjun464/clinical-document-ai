# Clinical Document Intelligence Platform

This portfolio project explores how a clinical document workflow might flag synthetic requisition records for manual review.

Important safety note: this project uses only synthetic data. It must not contain real patient data, employer documents, proprietary business rules, internal company information, scanned forms, OCR output, or protected health information.

## First milestone

This milestone creates a clean Python 3.11 project that generates a labeled synthetic dataset. The label, `requires_manual_review`, is created from simple synthetic validation rules so the dataset can later be used to train a scikit-learn model.

This milestone intentionally does not include FastAPI, LangChain, MLflow, Docker, cloud deployment, OCR, or deep learning.

## Project structure

```text
clinical-document-ai/
  data/raw/
  data/processed/
  notebooks/
  src/data/
  src/features/
  src/models/
  tests/
  docs/
  models/
```

## Setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate the synthetic dataset

```powershell
python src/data/generate_synthetic_data.py
```

The script writes:

```text
data/raw/synthetic_clinical_requisitions.csv
```

The generated file contains 1,000 synthetic records by default.

## Run tests

```powershell
pytest
```

## Synthetic manual-review rules

A requisition requires manual review when at least one of these synthetic rules is true:

1. `duplicate_kit` is `True`
2. `ocr_confidence` is less than `0.82`
3. `missing_field_count` is greater than or equal to `2`

These are learning rules for a portfolio project. They are not copied from any employer, lab, clinical system, or proprietary document workflow.

## Dataset columns

See [docs/data_dictionary.md](docs/data_dictionary.md) for a complete description of each field.
