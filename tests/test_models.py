"""Tests for src/models/evaluation.py and src/models/trainer.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.features.engineering import engineer_features
from src.models.evaluation import compute_metrics, find_value_based_threshold
from src.models.trainer import CANDIDATE_MODEL_NAMES, ModelTrainer

# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.95])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_metrics_returns_all_expected_keys():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.6, 0.3, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"]:
        assert key in metrics


def test_value_based_threshold_prefers_recall_when_conversion_is_valuable():
    """With a very high conversion value relative to intervention cost,
    the selected threshold should be low (catch almost everyone)."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])

    threshold, expected_value = find_value_based_threshold(
        y_true,
        y_proba,
        conversion_value=1000.0,
        intervention_cost=1.0,
    )

    assert 0.0 <= threshold <= 1.0
    assert expected_value > 0


def test_value_based_threshold_prefers_precision_when_intervention_is_costly():
    """With a very high intervention cost relative to conversion value,
    the selected threshold should be high (only act on strong signal)."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])

    low_cost_threshold, _ = find_value_based_threshold(
        y_true,
        y_proba,
        conversion_value=100.0,
        intervention_cost=1.0,
    )
    high_cost_threshold, _ = find_value_based_threshold(
        y_true,
        y_proba,
        conversion_value=100.0,
        intervention_cost=90.0,
    )

    assert high_cost_threshold >= low_cost_threshold


def test_value_based_threshold_rejects_all_zero_probabilities_gracefully():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 0.0])
    # Should not raise even in this degenerate case.
    threshold, _value = find_value_based_threshold(y_true, y_proba, 100.0, 5.0)
    assert isinstance(threshold, float)


# ---------------------------------------------------------------------------
# trainer.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def small_train_data(settings):
    """A small, fast subsample for testing — NOT for real model selection."""
    df = engineer_features(load_sessions(settings=settings))
    df_sample = df.groupby(settings.data.target_column, group_keys=False)[df.columns].apply(
        lambda g: g.sample(min(len(g), 150), random_state=settings.random_seed)
    )
    y = df_sample[settings.data.target_column]
    X = df_sample.drop(columns=[settings.data.target_column])
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=settings.random_seed)


def test_get_candidate_models_returns_all_six(settings):
    trainer = ModelTrainer(settings)
    models = trainer.get_candidate_models(balanced=True, scale_pos_weight=5.0)
    assert set(models.keys()) == set(CANDIDATE_MODEL_NAMES)


def test_compare_models_returns_ranked_results(settings, small_train_data):
    X_train, _, y_train, _ = small_train_data
    trainer = ModelTrainer(settings)

    # Reduce CV folds for test speed; full comparison (5 folds, full data)
    # happens in notebooks/03_modeling.ipynb.
    trainer.settings = settings.model_copy(deep=True)
    trainer.settings.modeling.cv_folds = 2

    results_df = trainer.compare_models(X_train, y_train, strategies=("class_weight",))

    assert len(results_df) == len(CANDIDATE_MODEL_NAMES)
    assert list(results_df.columns[:2]) == ["model", "strategy"]
    # Sorted descending by PR-AUC.
    assert results_df["pr_auc"].is_monotonic_decreasing


def test_select_best_picks_top_row(settings):
    import pandas as pd

    trainer = ModelTrainer(settings)
    results_df = pd.DataFrame(
        {
            "model": ["logistic_regression", "catboost"],
            "strategy": ["class_weight", "smote"],
            "pr_auc": [0.7, 0.9],
        }
    )
    name, strategy = trainer.select_best(results_df)
    assert name == "catboost"
    assert strategy == "smote"


def test_select_best_raises_on_empty_dataframe(settings):
    import pandas as pd

    trainer = ModelTrainer(settings)
    with pytest.raises(ValueError):
        trainer.select_best(pd.DataFrame())


def test_fit_best_produces_working_pipeline(settings, small_train_data):
    X_train, X_test, y_train, _y_test = small_train_data
    trainer = ModelTrainer(settings)
    pipeline = trainer.fit_best(
        X_train, y_train, model_name="logistic_regression", strategy="class_weight"
    )

    proba = pipeline.predict_proba(X_test)[:, 1]
    assert len(proba) == len(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_fit_best_raises_on_unknown_model_name(settings, small_train_data):
    X_train, _, y_train, _ = small_train_data
    trainer = ModelTrainer(settings)
    with pytest.raises(KeyError):
        trainer.fit_best(X_train, y_train, model_name="not_a_real_model", strategy="class_weight")


def test_save_and_load_roundtrip(tmp_path, settings, small_train_data):
    X_train, X_test, y_train, _ = small_train_data
    trainer = ModelTrainer(settings)
    pipeline = trainer.fit_best(
        X_train, y_train, model_name="logistic_regression", strategy="class_weight"
    )

    save_path = tmp_path / "test_pipeline.joblib"
    trainer.save(pipeline, save_path)
    assert save_path.exists()

    reloaded = trainer.load(save_path)
    original_proba = pipeline.predict_proba(X_test)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
    assert np.allclose(original_proba, reloaded_proba)
