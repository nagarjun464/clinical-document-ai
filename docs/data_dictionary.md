# Data Dictionary

This dataset is fully synthetic and is intended for portfolio learning only. It does not contain real patient data, employer documents, proprietary rules, clinical notes, scanned requisitions, or internal company information.

## Synthetic validation rules

The target column is `requires_manual_review`.

A record is labeled `True` when at least one of these synthetic rules is true:

1. `duplicate_kit` is `True`
2. `ocr_confidence` is less than `0.82`
3. `missing_field_count` is greater than or equal to `2`

Otherwise, `requires_manual_review` is `False`.

These rules are simple teaching rules. They are not real clinical, laboratory, compliance, employer, or operational rules.

## Columns

| Column | Type | Example | Description |
| --- | --- | --- | --- |
| `document_id` | string | `REQ-A1B2-12345678` | Synthetic unique identifier for a requisition record. This is not a real document ID. |
| `test_type` | string | `Complete Blood Count` | Synthetic category for the requested test. |
| `specimen_type` | string | `Blood` | Synthetic category for the specimen listed on the requisition. |
| `date_of_birth_present` | boolean | `True` | Whether a date-of-birth field is present. The actual date of birth is not stored. |
| `provider_present` | boolean | `True` | Whether provider information is present. No real provider identity is stored. |
| `collection_date_present` | boolean | `False` | Whether the collection-date field is present. The actual date is not stored. |
| `diagnosis_code_present` | boolean | `True` | Whether a diagnosis-code field is present. No real diagnosis information is stored. |
| `duplicate_kit` | boolean | `False` | Synthetic flag indicating whether a duplicate kit scenario was generated. |
| `ocr_confidence` | float | `0.934` | Synthetic OCR confidence score between `0.0` and `1.0`. This is simulated and does not come from OCR software. |
| `missing_field_count` | integer | `1` | Count of missing required field-presence indicators among date of birth, provider, collection date, and diagnosis code. |
| `requires_manual_review` | boolean | `True` | Rule-generated target label that indicates whether the synthetic record should be reviewed manually. |

## Allowed values

`test_type` values:

- `Complete Blood Count`
- `Comprehensive Metabolic Panel`
- `Lipid Panel`
- `Thyroid Panel`
- `Genetic Screening`
- `Infectious Disease Panel`

`specimen_type` values:

- `Blood`
- `Saliva`
- `Urine`
- `Buccal Swab`
- `Tissue`
