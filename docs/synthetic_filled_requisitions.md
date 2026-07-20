# Synthetic Filled Requisition PDFs

This milestone generates multiple filled requisition PDFs using fake test data only.

The provided blank requisition PDFs are used as visual templates. The generated PDFs do not contain real patient data, employer documents, proprietary business rules, or internal company information.

## What This Step Does

The generator:

1. Accepts one or more blank PDF templates.
2. Creates fake synthetic requisition records with Faker.
3. Overlays fake values onto page one of each template.
4. Saves filled synthetic PDFs under `output/pdf/synthetic_requisitions/`.
5. Saves a ground-truth CSV under `data/processed/synthetic_filled_requisition_ground_truth.csv`.

This is not OCR and it is not real form extraction. It creates fake filled forms that can be used in the next milestone to test extraction logic.

## How To Run

From the project root:

```powershell
python -m src.data.generate_synthetic_filled_pdfs `
  --signatera-template "C:\Users\smart\OneDrive\Desktop\Desktop\SIgnatera.pdf" `
  --empower-template "C:\Users\smart\OneDrive\Desktop\Desktop\Empower.pdf" `
  --womens-health-template "C:\Users\smart\OneDrive\Desktop\Womens-Health-Clinical-UNIV-14-Panel-4-Sample-Collection-Form.pdf" `
  --records-per-template 3
```

With three templates and three records per template, the script creates nine filled synthetic PDFs.

## Outputs

Generated PDFs:

```text
output/pdf/synthetic_requisitions/
```

Ground-truth CSV:

```text
data/processed/synthetic_filled_requisition_ground_truth.csv
```

These are generated artifacts and are ignored by Git.

## Ground-Truth Fields

The ground-truth CSV includes the fake values inserted into each generated PDF:

- synthetic requisition ID
- form type
- output PDF path
- template PDF name
- fake patient first and last name
- fake date of birth
- fake phone and email
- fake address
- fake biological sex
- fake collection date
- fake clinician and clinic details
- fake test ordered
- fake specimen type
- fake diagnosis code
- fake payment type
- synthetic required-field presence flags

## Current Limitations

- The PDFs are filled by overlaying digital text on top of blank templates.
- Only selected page-one fields are filled.
- The layout coordinates are template-specific.
- This step does not extract data from PDFs yet.
- This step does not perform OCR.
- More realistic noise, such as skew, blur, handwriting, and extraction confidence, is intentionally left for a later milestone.

## Next Step

After approval, the next milestone should extract values from these generated synthetic PDFs and compare the extracted values with the ground-truth CSV.
