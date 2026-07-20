from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.data.generate_synthetic_data import generate_synthetic_clinical_requisitions
from src.models.train import (
    build_model_pipelines,
    select_best_model,
    train_models,
)


def test_build_model_pipelines_creates_expected_models() -> None:
    pipelines = build_model_pipelines(random_state=42)

    assert sorted(pipelines) == ["Logistic Regression", "Random Forest"]
    assert all(isinstance(pipeline, Pipeline) for pipeline in pipelines.values())


def test_train_models_returns_metrics_and_saves_selected_model(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    model_path = tmp_path / "manual_review_model.joblib"
    df = generate_synthetic_clinical_requisitions(record_count=250, random_seed=204)
    df.to_csv(data_path, index=False)

    result = train_models(
        data_path=data_path,
        model_output_path=model_path,
        random_state=42,
        test_size=0.25,
    )

    assert result.train_rows == 187
    assert result.test_rows == 63
    assert result.logistic_regression.metrics.recall >= 0.0
    assert result.random_forest.metrics.roc_auc >= 0.0
    assert result.selected_model.name in {"Logistic Regression", "Random Forest"}
    assert model_path.exists()

    saved_package = joblib.load(model_path)
    assert saved_package["model_name"] == result.selected_model.name
    assert saved_package["target_column"] == "requires_manual_review"
    assert "document_id" not in saved_package["feature_columns"]


def test_select_best_model_prefers_higher_f1_score(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    model_path = tmp_path / "manual_review_model.joblib"
    df = generate_synthetic_clinical_requisitions(record_count=200, random_seed=205)
    df.to_csv(data_path, index=False)
    result = train_models(data_path=data_path, model_output_path=model_path)

    selected = select_best_model([result.logistic_regression, result.random_forest])
    scores = [
        result.logistic_regression.metrics.f1_score,
        result.random_forest.metrics.f1_score,
    ]

    assert selected.metrics.f1_score == max(scores)
