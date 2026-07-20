from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from src.data.generate_synthetic_filled_pdfs import (
    TemplateSpec,
    generate_synthetic_filled_requisitions,
    validate_template_specs,
)


def create_blank_template(path: Path) -> None:
    """Create a simple two-page PDF template for tests."""
    pdf_canvas = canvas.Canvas(str(path), pagesize=(612, 792))
    pdf_canvas.drawString(72, 720, "Synthetic Blank Template")
    pdf_canvas.showPage()
    pdf_canvas.drawString(72, 720, "Synthetic Blank Template Page 2")
    pdf_canvas.save()


def test_generate_synthetic_filled_requisitions_creates_pdfs_and_ground_truth(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "template.pdf"
    output_dir = tmp_path / "filled_pdfs"
    ground_truth_path = tmp_path / "ground_truth.csv"
    create_blank_template(template_path)

    ground_truth_df = generate_synthetic_filled_requisitions(
        template_specs=[TemplateSpec("signatera", template_path)],
        records_per_template=2,
        output_dir=output_dir,
        ground_truth_path=ground_truth_path,
        random_seed=301,
    )

    assert len(ground_truth_df) == 2
    assert ground_truth_path.exists()
    assert sorted(output_dir.glob("*.pdf"))

    saved_ground_truth = pd.read_csv(ground_truth_path)
    assert len(saved_ground_truth) == 2
    assert saved_ground_truth["form_type"].tolist() == ["signatera", "signatera"]
    assert saved_ground_truth["date_of_birth_present"].all()
    assert saved_ground_truth["provider_present"].all()

    first_pdf = Path(saved_ground_truth.loc[0, "output_pdf"])
    reader = PdfReader(str(first_pdf))
    assert len(reader.pages) == 2


def test_generation_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    template_path = tmp_path / "template.pdf"
    create_blank_template(template_path)

    first = generate_synthetic_filled_requisitions(
        template_specs=[TemplateSpec("empower", template_path)],
        records_per_template=1,
        output_dir=tmp_path / "first",
        ground_truth_path=tmp_path / "first.csv",
        random_seed=302,
    )
    second = generate_synthetic_filled_requisitions(
        template_specs=[TemplateSpec("empower", template_path)],
        records_per_template=1,
        output_dir=tmp_path / "second",
        ground_truth_path=tmp_path / "second.csv",
        random_seed=302,
    )

    comparable_columns = [
        column for column in first.columns if column not in {"output_pdf"}
    ]
    pd.testing.assert_frame_equal(
        first[comparable_columns],
        second[comparable_columns],
    )


def test_validate_template_specs_rejects_missing_template(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_template_specs([TemplateSpec("signatera", tmp_path / "missing.pdf")])


def test_validate_template_specs_rejects_duplicate_form_type(tmp_path: Path) -> None:
    template_path = tmp_path / "template.pdf"
    create_blank_template(template_path)

    with pytest.raises(ValueError, match="Duplicate template"):
        validate_template_specs(
            [
                TemplateSpec("womens_health", template_path),
                TemplateSpec("womens_health", template_path),
            ]
        )
