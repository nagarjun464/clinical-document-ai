# Baseline Model Training

This milestone trains the first machine-learning models for the synthetic Clinical Document Intelligence Platform dataset.

The goal is to predict whether a synthetic requisition requires manual review:

```text
requires_manual_review = True
```

No real patient data, employer documents, proprietary business rules, OCR output, or internal company information are used.

## Models

Two scikit-learn models are compared:

1. Logistic Regression baseline
2. Random Forest comparison model

Both models use a scikit-learn `Pipeline` with a `ColumnTransformer`.

## Features Used

The model uses fields that would be available before a manual-review decision in this synthetic workflow:

- `test_type`
- `specimen_type`
- `date_of_birth_present`
- `provider_present`
- `collection_date_present`
- `diagnosis_code_present`
- `duplicate_kit`
- `ocr_confidence`
- `missing_field_count`

## Features Excluded

These fields are excluded to reduce leakage risk:

- `document_id`
- `requires_manual_review`
- `patient_name` if such a field is ever added later

`document_id` is only an identifier and should not teach the model anything useful. `requires_manual_review` is the answer the model is trying to predict, so including it would be direct leakage.

## Evaluation Metrics

The training pipeline evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The metrics focus on the positive class: `requires_manual_review=True`.

## Baseline Results

The first run used an 80/20 stratified train/test split:

- Training rows: 800
- Test rows: 200

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.950 | 0.727 | 0.960 | 0.828 | 0.993 | `[[166, 9], [1, 24]]` |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `[[175, 0], [0, 25]]` |

The selected model is Random Forest because it had the strongest F1-score, recall, and ROC-AUC on the test set.

The confusion matrix format is:

```text
[[true negatives, false positives],
 [false negatives, true positives]]
```

## Why Recall Matters

Recall answers this question:

```text
Of the requisitions that truly require manual review, how many did the model catch?
```

For this use case, recall matters because a false negative means the model says a requisition does not need review even though it actually does. In a real workflow, that could allow an incomplete or problematic requisition to move forward without manual attention.

## False Positives And False Negatives

A false positive is a requisition that does not require review but the model flags for review. This creates extra manual work.

A false negative is a requisition that requires review but the model misses. This is riskier because the workflow may fail to catch a record that should be checked.

In this run:

- Logistic Regression had 9 false positives and 1 false negative.
- Random Forest had 0 false positives and 0 false negatives.

## Data Leakage Risks

The biggest leakage risk is accidentally training on fields that reveal the answer. The pipeline excludes `requires_manual_review` and `document_id`.

There is also a synthetic-data limitation: the target label is generated from simple rules involving `duplicate_kit`, `ocr_confidence`, and `missing_field_count`. Because those same validation-style fields are used as model inputs, the models can learn the synthetic rule very well. This is acceptable for a first portfolio baseline, but model performance should not be interpreted as real-world clinical performance.

## Synthetic Dataset Limitations

This dataset is intentionally simple:

- It has only 1,000 synthetic records.
- It does not contain real forms or real patient data.
- It does not include OCR extraction errors from actual scanned documents.
- The target label is generated from transparent synthetic rules.
- The model is learning a simulated workflow, not real clinical decision-making.

Future milestones can add richer synthetic filled forms, more realistic extraction noise, stronger validation checks, and more careful model evaluation.

## How To Run

From the project root:

```powershell
python -m src.models.train
```

The selected model is saved locally to:

```text
models/manual_review_model.joblib
```

Model artifacts are ignored by Git because they are generated outputs.
