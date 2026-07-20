"""Train baseline models for synthetic manual-review prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.generate_synthetic_data import DEFAULT_OUTPUT_PATH
from src.data.validate_dataset import load_dataset, validate_dataset
from src.features.build_features import (
    FEATURE_COLUMNS,
    build_preprocessor,
    split_features_and_target,
)
from src.models.evaluate import (
    ClassificationMetrics,
    calculate_classification_metrics,
    format_metrics,
    get_positive_class_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "manual_review_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass(frozen=True)
class TrainedModelResult:
    """Training and evaluation result for one fitted model."""

    name: str
    pipeline: Pipeline
    metrics: ClassificationMetrics


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary of the full model training run."""

    logistic_regression: TrainedModelResult
    random_forest: TrainedModelResult
    selected_model: TrainedModelResult
    model_output_path: Path
    train_rows: int
    test_rows: int
    feature_columns: list[str]


def build_model_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Create the baseline and comparison model pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def fit_and_evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> TrainedModelResult:
    """Fit one model pipeline and calculate test-set metrics."""
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = get_positive_class_scores(pipeline, X_test)
    metrics = calculate_classification_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_score=y_score,
    )

    return TrainedModelResult(name=name, pipeline=pipeline, metrics=metrics)


def select_best_model(results: list[TrainedModelResult]) -> TrainedModelResult:
    """Select the best model using F1-score, then recall, then ROC-AUC."""
    if not results:
        raise ValueError("At least one trained model result is required.")

    return max(
        results,
        key=lambda result: (
            result.metrics.f1_score,
            result.metrics.recall,
            result.metrics.roc_auc,
        ),
    )


def save_model(
    result: TrainedModelResult,
    output_path: str | Path = DEFAULT_MODEL_OUTPUT_PATH,
    feature_columns: list[str] | None = None,
) -> Path:
    """Save the selected fitted model and metadata using joblib."""
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model_name": result.name,
        "pipeline": result.pipeline,
        "metrics": result.metrics,
        "feature_columns": feature_columns or FEATURE_COLUMNS,
        "target_column": "requires_manual_review",
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_package, model_path)
    return model_path


def train_models(
    data_path: str | Path = DEFAULT_OUTPUT_PATH,
    model_output_path: str | Path = DEFAULT_MODEL_OUTPUT_PATH,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> TrainingRunResult:
    """Train and compare baseline models on the synthetic requisition dataset."""
    df = load_dataset(data_path)
    validation_result = validate_dataset(df, expected_count=None, path=data_path)
    if not validation_result.is_valid:
        error_text = "; ".join(validation_result.errors)
        raise ValueError(f"Dataset validation failed before training: {error_text}")

    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipelines = build_model_pipelines(random_state=random_state)
    trained_results = [
        fit_and_evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        for name, pipeline in pipelines.items()
    ]
    selected_model = select_best_model(trained_results)
    saved_model_path = save_model(
        selected_model,
        output_path=model_output_path,
        feature_columns=list(X.columns),
    )

    result_by_name = {result.name: result for result in trained_results}
    return TrainingRunResult(
        logistic_regression=result_by_name["Logistic Regression"],
        random_forest=result_by_name["Random Forest"],
        selected_model=selected_model,
        model_output_path=saved_model_path,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_columns=list(X.columns),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train baseline models for synthetic manual-review prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Input synthetic dataset CSV. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help=f"Path for the selected joblib model. Default: {DEFAULT_MODEL_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the model training workflow from the command line."""
    args = parse_args()
    result = train_models(data_path=args.input, model_output_path=args.model_output)

    print(f"Training rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(format_metrics(result.logistic_regression.name, result.logistic_regression.metrics))
    print(format_metrics(result.random_forest.name, result.random_forest.metrics))
    print(f"Selected model: {result.selected_model.name}")
    print(f"Saved selected model to: {result.model_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
