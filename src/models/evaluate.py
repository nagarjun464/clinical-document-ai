"""Evaluate manual-review classification models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


POSITIVE_LABEL = True


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics focused on the requires_manual_review=True class."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    @property
    def true_negatives(self) -> int:
        """Return the number of correctly predicted no-review records."""
        return self.confusion_matrix[0][0]

    @property
    def false_positives(self) -> int:
        """Return no-review records incorrectly predicted as manual review."""
        return self.confusion_matrix[0][1]

    @property
    def false_negatives(self) -> int:
        """Return manual-review records incorrectly predicted as no review."""
        return self.confusion_matrix[1][0]

    @property
    def true_positives(self) -> int:
        """Return correctly predicted manual-review records."""
        return self.confusion_matrix[1][1]


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> ClassificationMetrics:
    """Calculate classification metrics for the manual-review class."""
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        confusion_matrix=matrix.astype(int).tolist(),
    )


def get_positive_class_scores(model: object, X: object) -> np.ndarray:
    """Return predicted probabilities for the requires_manual_review=True class."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must support predict_proba to calculate ROC-AUC.")

    classes = list(model.classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError("Model classes do not include the positive manual-review label.")

    positive_class_index = classes.index(POSITIVE_LABEL)
    return model.predict_proba(X)[:, positive_class_index]


def format_metrics(name: str, metrics: ClassificationMetrics) -> str:
    """Format model metrics for readable command-line output."""
    return (
        f"{name}\n"
        f"  Accuracy:  {metrics.accuracy:.3f}\n"
        f"  Precision: {metrics.precision:.3f}\n"
        f"  Recall:    {metrics.recall:.3f}\n"
        f"  F1-score:  {metrics.f1_score:.3f}\n"
        f"  ROC-AUC:   {metrics.roc_auc:.3f}\n"
        f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics.confusion_matrix}"
    )
