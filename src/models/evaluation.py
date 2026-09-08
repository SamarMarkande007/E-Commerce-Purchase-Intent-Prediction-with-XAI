"""Evaluation: the full metric suite and value-based threshold selection.

Per the brief: accuracy is a trap on this ~16%-positive dataset (a
model predicting "no purchase" for everyone scores ~84% accuracy while
catching zero buyers). Every function here treats PR-AUC and recall on
the positive class as the metrics that matter, and reports accuracy
only for completeness — never as the headline number.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

#: Maps our metric names to sklearn's cross_validate scoring strings.
#: Used by src/models/trainer.py's cross-validation comparison.
SKLEARN_SCORING: dict[str, str] = {
    "pr_auc": "average_precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "accuracy": "accuracy",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute the full evaluation metric suite for one set of predictions.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1), i.e. proba thresholded.
        y_proba: Predicted probability of the positive class.

    Returns:
        A dict with keys: accuracy, precision, recall, f1, roc_auc,
        pr_auc, confusion_matrix (as a nested list [[tn, fp], [fn, tp]]).
        precision/recall/f1 are computed for the positive (buying) class.
    """
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
    }


def find_value_based_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    conversion_value: float,
    intervention_cost: float,
) -> tuple[float, float]:
    """Choose a decision threshold from business value, not 0.5.

    At each candidate threshold, computes the expected value of acting
    on that threshold: ``value = (true positives * conversion_value) -
    (false positives * intervention_cost)``. The threshold maximising
    this expected value is returned. This is the same formula used and
    explained in notebooks/03_modeling.ipynb §5 — intervention cost is
    charged only against false positives (sessions flagged that would
    not have converted anyway), not against every flagged session,
    since a true positive's intervention is presumed to have paid for
    itself via the captured conversion.

    This models a scenario where a positive prediction triggers some
    intervention (e.g. a discount nudge) that costs ``intervention_cost``
    but, if it correctly targets a converting session, captures
    ``conversion_value``.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_proba: Predicted probability of the positive class.
        conversion_value: Business value of correctly catching one
            converting session.
        intervention_cost: Cost of triggering the intervention on one
            predicted-positive session that would NOT have converted.

    Returns:
        A tuple ``(best_threshold, best_expected_value)``.
    """
    y_true = np.asarray(y_true)
    _, _, thresholds = precision_recall_curve(y_true, y_proba)

    expected_value = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        expected_value[i] = tp * conversion_value - fp * intervention_cost

    best_idx = int(np.argmax(expected_value))
    best_threshold = float(thresholds[best_idx])
    best_value = float(expected_value[best_idx])

    logger.info(
        "Value-based threshold selected: %.4f (expected value=%.2f, "
        "vs conversion_value=%.2f, intervention_cost=%.2f)",
        best_threshold, best_value, conversion_value, intervention_cost,
    )

    return best_threshold, best_value
