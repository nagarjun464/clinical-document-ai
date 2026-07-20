"""Generate synthetic filled requisition PDFs from blank PDF templates.

The generated requisitions contain fake data only. They are intended for
portfolio development and extraction testing, not clinical use.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from faker import Faker
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import black, white
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf" / "synthetic_requisitions"
DEFAULT_GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "processed" / "synthetic_filled_requisition_ground_truth.csv"
)
DEFAULT_RECORDS_PER_TEMPLATE = 3
DEFAULT_RANDOM_SEED = 42

FORM_TYPES = ("signatera", "empower", "womens_health")
PAYMENT_TYPES = ("Bill Insurance", "Bill Clinic", "Self Pay")
BIOLOGICAL_SEXES = ("F", "M")

TEST_BY_FORM_TYPE = {
    "signatera": ("Signatera", "Altera", "Empower"),
    "empower": ("BRCA1 & BRCA2", "Lynch Syndrome", "Multi-Cancer Panel"),
    "womens_health": ("Panorama", "Horizon", "Vistara"),
}

SPECIMEN_BY_FORM_TYPE = {
    "signatera": ("Blood", "Tissue"),
    "empower": ("Blood", "Saliva"),
    "womens_health": ("Blood", "Buccal Swab"),
}

DIAGNOSIS_CODES = {
    "signatera": ("Z85.038", "Z85.3", "Z85.43"),
    "empower": ("Z80.3", "Z80.41", "Z15.01"),
    "womens_health": ("Z36.0", "Z31.438", "O09.511"),
}


@dataclass(frozen=True)
class TemplateSpec:
    """PDF template path and form type."""

    form_type: str
    path: Path


@dataclass(frozen=True)
class SyntheticFilledRequisition:
    """Ground-truth values inserted into one synthetic filled requisition."""

    synthetic_requisition_id: str
    form_type: str
    output_pdf: str
    template_pdf_name: str
    patient_first_name: str
    patient_last_name: str
    date_of_birth: str
    cell_phone: str
    patient_email: str
    address: str
    city: str
    state: str
    zip_code: str
    biological_sex: str
    collection_date: str
    clinician_name: str
    clinic_or_organization: str
    clinician_phone: str
    test_ordered: str
    specimen_type: str
    diagnosis_code: str
    payment_type: str
    duplicate_kit: bool
    date_of_birth_present: bool
    provider_present: bool
    collection_date_present: bool
    diagnosis_code_present: bool


def validate_template_specs(template_specs: list[TemplateSpec]) -> None:
    """Validate template definitions before generation starts."""
    if not template_specs:
        raise ValueError("At least one template PDF is required.")

    seen_form_types: set[str] = set()
    for spec in template_specs:
        if spec.form_type not in FORM_TYPES:
            raise ValueError(f"Unsupported form_type: {spec.form_type}")
        if spec.form_type in seen_form_types:
            raise ValueError(f"Duplicate template for form_type: {spec.form_type}")
        if not spec.path.exists():
            raise FileNotFoundError(f"Template PDF does not exist: {spec.path}")
        if spec.path.suffix.lower() != ".pdf":
            raise ValueError(f"Template path must be a PDF: {spec.path}")
        seen_form_types.add(spec.form_type)


def build_synthetic_record(
    fake: Faker,
    form_type: str,
    sequence_number: int,
    output_pdf: Path,
    template_pdf_name: str,
) -> SyntheticFilledRequisition:
    """Create fake ground-truth values for one requisition."""
    synthetic_requisition_id = f"SYN-{form_type.upper().replace('_', '-')}-{sequence_number:04d}"
    state = fake.state_abbr()
    first_name = fake.first_name()
    last_name = fake.last_name()

    return SyntheticFilledRequisition(
        synthetic_requisition_id=synthetic_requisition_id,
        form_type=form_type,
        output_pdf=str(output_pdf),
        template_pdf_name=template_pdf_name,
        patient_first_name=first_name,
        patient_last_name=last_name,
        date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%m/%d/%Y"),
        cell_phone=fake.numerify("(###) ###-####"),
        patient_email=f"{first_name.lower()}.{last_name.lower()}@synthetic.example",
        address=fake.street_address(),
        city=fake.city(),
        state=state,
        zip_code=fake.postcode(),
        biological_sex=fake.random_element(BIOLOGICAL_SEXES),
        collection_date=fake.date_between(start_date="-30d", end_date="today").strftime("%m/%d/%Y"),
        clinician_name=f"Dr. {fake.first_name()} {fake.last_name()}",
        clinic_or_organization=f"{fake.last_name()} Synthetic Clinic",
        clinician_phone=fake.numerify("(###) ###-####"),
        test_ordered=fake.random_element(TEST_BY_FORM_TYPE[form_type]),
        specimen_type=fake.random_element(SPECIMEN_BY_FORM_TYPE[form_type]),
        diagnosis_code=fake.random_element(DIAGNOSIS_CODES[form_type]),
        payment_type=fake.random_element(PAYMENT_TYPES),
        duplicate_kit=False,
        date_of_birth_present=True,
        provider_present=True,
        collection_date_present=True,
        diagnosis_code_present=True,
    )


def draw_text(
    pdf_canvas: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    font_size: int = 8,
    max_chars: int = 34,
) -> None:
    """Draw clipped synthetic text on a PDF overlay."""
    clipped_value = value[:max_chars]
    pdf_canvas.setFillColor(black)
    pdf_canvas.setFont("Helvetica", font_size)
    pdf_canvas.drawString(x, y, clipped_value)


def draw_check(pdf_canvas: canvas.Canvas, x: float, y: float) -> None:
    """Draw a simple checkbox mark on a PDF overlay."""
    pdf_canvas.setFillColor(black)
    pdf_canvas.setFont("Helvetica-Bold", 8)
    pdf_canvas.drawString(x, y, "X")


def draw_white_box(
    pdf_canvas: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Cover a template sample barcode placeholder with a synthetic text area."""
    pdf_canvas.setFillColor(white)
    pdf_canvas.setStrokeColor(white)
    pdf_canvas.rect(x, y, width, height, fill=True, stroke=False)


def draw_signatera_page_one(
    pdf_canvas: canvas.Canvas,
    record: SyntheticFilledRequisition,
) -> None:
    """Draw synthetic values on page one of the Signatera template."""
    draw_text(pdf_canvas, 346, 757, record.collection_date, 9, 10)
    draw_text(pdf_canvas, 132, 704, record.patient_first_name, 8, 24)
    draw_text(pdf_canvas, 294, 704, record.patient_last_name, 8, 24)
    draw_text(pdf_canvas, 441, 704, record.date_of_birth, 8, 10)
    draw_text(pdf_canvas, 548, 704, record.state, 8, 2)
    draw_text(pdf_canvas, 132, 674, "Synthetic Coordinator", 8, 28)
    draw_text(pdf_canvas, 294, 674, record.clinic_or_organization, 8, 28)
    draw_text(pdf_canvas, 460, 674, record.clinician_phone, 8, 15)
    draw_text(pdf_canvas, 143, 650, record.clinician_name, 8, 28)
    draw_text(pdf_canvas, 365, 563, record.clinician_name, 8, 28)
    draw_text(pdf_canvas, 538, 563, record.collection_date, 8, 10)

    if record.test_ordered == "Signatera":
        draw_check(pdf_canvas, 130, 476)
    elif record.test_ordered == "Altera":
        draw_check(pdf_canvas, 130, 318)
    else:
        draw_check(pdf_canvas, 130, 233)

    draw_check(pdf_canvas, 130, 112)
    draw_text(pdf_canvas, 386, 112, record.diagnosis_code, 8, 12)


def draw_empower_page_one(
    pdf_canvas: canvas.Canvas,
    record: SyntheticFilledRequisition,
) -> None:
    """Draw synthetic values on page one of the Empower template."""
    draw_text(pdf_canvas, 226, 748, record.collection_date, 10, 10)
    draw_text(pdf_canvas, 22, 660, record.patient_last_name, 8, 22)
    draw_text(pdf_canvas, 185, 660, record.patient_first_name, 8, 22)
    draw_text(pdf_canvas, 22, 638, record.date_of_birth, 8, 10)
    draw_check(pdf_canvas, 137 if record.biological_sex == "F" else 151, 635)
    draw_text(pdf_canvas, 232, 638, record.synthetic_requisition_id[-8:], 8, 12)
    draw_text(pdf_canvas, 354, 638, record.cell_phone, 8, 15)
    draw_text(pdf_canvas, 22, 617, record.patient_email, 7, 44)
    draw_text(pdf_canvas, 22, 595, record.address, 8, 34)
    draw_text(pdf_canvas, 22, 572, record.city, 8, 24)
    draw_text(pdf_canvas, 204, 572, record.state, 8, 2)
    draw_text(pdf_canvas, 261, 572, record.zip_code, 8, 10)
    draw_text(pdf_canvas, 320, 660, record.clinic_or_organization, 8, 34)
    draw_check(pdf_canvas, 320, 610)
    draw_text(pdf_canvas, 320, 494, "100 Synthetic Way", 8, 28)
    draw_text(pdf_canvas, 486, 494, record.city, 8, 20)
    draw_text(pdf_canvas, 320, 472, record.state, 8, 2)
    draw_text(pdf_canvas, 372, 472, record.zip_code, 8, 10)
    draw_text(pdf_canvas, 465, 472, record.clinician_phone, 8, 15)
    draw_text(pdf_canvas, 22, 453, record.clinician_name, 8, 28)
    draw_text(pdf_canvas, 238, 453, record.collection_date, 8, 10)

    if record.payment_type == "Bill Insurance":
        draw_check(pdf_canvas, 53, 388)
    elif record.payment_type == "Self Pay":
        draw_check(pdf_canvas, 120, 388)
    else:
        draw_check(pdf_canvas, 137, 377)

    if record.test_ordered == "BRCA1 & BRCA2":
        draw_check(pdf_canvas, 44, 305)
    elif record.test_ordered == "Lynch Syndrome":
        draw_check(pdf_canvas, 44, 282)
    else:
        draw_check(pdf_canvas, 385, 305)
    draw_text(pdf_canvas, 72, 75, f"Synthetic ICD-10: {record.diagnosis_code}", 7, 28)


def draw_womens_health_page_one(
    pdf_canvas: canvas.Canvas,
    record: SyntheticFilledRequisition,
) -> None:
    """Draw synthetic values on page one of the women's health template."""
    draw_text(pdf_canvas, 190, 756, record.collection_date, 9, 10)
    draw_text(pdf_canvas, 20, 719, record.patient_last_name, 8, 24)
    draw_text(pdf_canvas, 180, 719, record.patient_first_name, 8, 24)
    draw_text(pdf_canvas, 20, 696, record.date_of_birth, 8, 10)
    draw_text(pdf_canvas, 180, 696, record.cell_phone, 8, 15)
    draw_text(pdf_canvas, 20, 674, record.patient_email, 7, 48)
    draw_text(pdf_canvas, 20, 650, record.address, 8, 48)
    draw_text(pdf_canvas, 20, 626, record.city, 8, 24)
    draw_text(pdf_canvas, 178, 626, record.state, 8, 2)
    draw_text(pdf_canvas, 225, 626, record.zip_code, 8, 10)
    draw_check(pdf_canvas, 276 if record.biological_sex == "F" else 302, 633)
    draw_text(pdf_canvas, 338, 704, record.clinic_or_organization, 8, 42)
    draw_check(pdf_canvas, 338, 660)
    draw_text(pdf_canvas, 508, 686, record.clinician_phone, 8, 15)
    draw_text(pdf_canvas, 338, 503, record.clinician_name, 8, 32)

    if record.payment_type == "Bill Insurance":
        draw_check(pdf_canvas, 19, 607)
    elif record.payment_type == "Bill Clinic":
        draw_check(pdf_canvas, 88, 607)
    else:
        draw_check(pdf_canvas, 288, 607)

    if record.test_ordered == "Panorama":
        draw_check(pdf_canvas, 34, 302)
    elif record.test_ordered == "Horizon":
        draw_check(pdf_canvas, 338, 360)
    else:
        draw_check(pdf_canvas, 82, 164)

    draw_check(pdf_canvas, 20, 240)
    draw_text(pdf_canvas, 52, 240, record.diagnosis_code, 7, 10)
    draw_text(pdf_canvas, 400, 91, record.clinician_name, 8, 28)
    draw_text(pdf_canvas, 584, 91, record.collection_date, 8, 10)


def draw_overlay_page(
    pdf_canvas: canvas.Canvas,
    record: SyntheticFilledRequisition,
    page_number: int,
) -> None:
    """Draw synthetic values for a template page."""
    if page_number != 0:
        return

    if record.form_type == "signatera":
        draw_signatera_page_one(pdf_canvas, record)
    elif record.form_type == "empower":
        draw_empower_page_one(pdf_canvas, record)
    elif record.form_type == "womens_health":
        draw_womens_health_page_one(pdf_canvas, record)
    else:
        raise ValueError(f"Unsupported form_type: {record.form_type}")


def create_overlay_pdf(
    record: SyntheticFilledRequisition,
    template_reader: PdfReader,
    overlay_path: Path,
) -> None:
    """Create a transparent PDF overlay containing synthetic filled values."""
    first_page = template_reader.pages[0]
    width = float(first_page.mediabox.width)
    height = float(first_page.mediabox.height)

    pdf_canvas = canvas.Canvas(str(overlay_path), pagesize=(width, height))
    pdf_canvas.setAuthor("Synthetic Clinical Document Intelligence Platform")
    pdf_canvas.setTitle(record.synthetic_requisition_id)

    for page_number, _page in enumerate(template_reader.pages):
        draw_overlay_page(pdf_canvas, record, page_number=page_number)
        pdf_canvas.showPage()

    pdf_canvas.save()


def fill_pdf_template(
    template_path: Path,
    output_path: Path,
    record: SyntheticFilledRequisition,
) -> None:
    """Merge synthetic values onto a blank PDF template."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_reader = PdfReader(str(template_path))

    with tempfile.TemporaryDirectory() as temporary_directory:
        overlay_path = Path(temporary_directory) / "overlay.pdf"
        create_overlay_pdf(record, template_reader, overlay_path)
        overlay_reader = PdfReader(str(overlay_path))
        writer = PdfWriter(clone_from=str(template_path))

        for page_number, page in enumerate(writer.pages):
            page.merge_page(overlay_reader.pages[page_number])

        with output_path.open("wb") as output_file:
            writer.write(output_file)


def generate_synthetic_filled_requisitions(
    template_specs: list[TemplateSpec],
    records_per_template: int = DEFAULT_RECORDS_PER_TEMPLATE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic filled requisition PDFs and a ground-truth CSV."""
    validate_template_specs(template_specs)
    if records_per_template <= 0:
        raise ValueError("records_per_template must be greater than zero.")

    output_directory = Path(output_dir)
    ground_truth_file = Path(ground_truth_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    ground_truth_file.parent.mkdir(parents=True, exist_ok=True)

    Faker.seed(random_seed)
    fake = Faker("en_US")
    fake.seed_instance(random_seed)

    records: list[SyntheticFilledRequisition] = []
    sequence_number = 1
    for spec in template_specs:
        for _ in range(records_per_template):
            output_pdf = output_directory / f"{spec.form_type}_{sequence_number:04d}.pdf"
            record = build_synthetic_record(
                fake=fake,
                form_type=spec.form_type,
                sequence_number=sequence_number,
                output_pdf=output_pdf,
                template_pdf_name=spec.path.name,
            )
            fill_pdf_template(
                template_path=spec.path,
                output_path=output_pdf,
                record=record,
            )
            records.append(record)
            sequence_number += 1

    ground_truth_df = pd.DataFrame([asdict(record) for record in records])
    ground_truth_df.to_csv(ground_truth_file, index=False)
    return ground_truth_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for synthetic filled PDF generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic filled requisition PDFs from blank templates."
    )
    parser.add_argument("--signatera-template", type=Path, help="Path to Signatera PDF template.")
    parser.add_argument("--empower-template", type=Path, help="Path to Empower PDF template.")
    parser.add_argument(
        "--womens-health-template",
        type=Path,
        help="Path to women's health PDF template.",
    )
    parser.add_argument(
        "--records-per-template",
        type=int,
        default=DEFAULT_RECORDS_PER_TEMPLATE,
        help=f"Number of synthetic PDFs to create per template. Default: {DEFAULT_RECORDS_PER_TEMPLATE}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated PDFs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help=f"Ground-truth CSV path. Default: {DEFAULT_GROUND_TRUTH_PATH}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible fake records. Default: {DEFAULT_RANDOM_SEED}.",
    )
    return parser.parse_args()


def build_template_specs_from_args(args: argparse.Namespace) -> list[TemplateSpec]:
    """Build template specs from provided command-line arguments."""
    template_specs: list[TemplateSpec] = []
    if args.signatera_template:
        template_specs.append(TemplateSpec("signatera", args.signatera_template))
    if args.empower_template:
        template_specs.append(TemplateSpec("empower", args.empower_template))
    if args.womens_health_template:
        template_specs.append(TemplateSpec("womens_health", args.womens_health_template))
    return template_specs


def main() -> int:
    """Run synthetic filled PDF generation from the command line."""
    args = parse_args()
    template_specs = build_template_specs_from_args(args)
    ground_truth_df = generate_synthetic_filled_requisitions(
        template_specs=template_specs,
        records_per_template=args.records_per_template,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        random_seed=args.seed,
    )

    print(f"Generated {len(ground_truth_df)} synthetic filled requisition PDFs.")
    print(f"PDF output directory: {args.output_dir}")
    print(f"Ground-truth CSV: {args.ground_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
