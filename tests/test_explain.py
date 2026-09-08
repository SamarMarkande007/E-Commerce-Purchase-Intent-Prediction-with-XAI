"""Tests for src/explain/shap_utils.py."""

from __future__ import annotations

import joblib
import pytest

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.explain.shap_utils import SessionExplainer
from src.features.engineering import engineer_features


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def pipeline(settings):
    path = settings.paths.models_dir / "best_pipeline.joblib"
    if not path.exists():
        pytest.skip("models/best_pipeline.joblib not found; run notebooks/03_modeling.ipynb first")
    return joblib.load(path)


@pytest.fixture(scope="module")
def sample_row(settings):
    df = engineer_features(load_sessions(settings=settings))
    X = df.drop(columns=[settings.data.target_column])
    return X.iloc[[0]]


def test_explainer_builds_from_pipeline(pipeline):
    explainer = SessionExplainer(pipeline)
    assert len(explainer.feature_names) > 0
    assert "PageValues" in explainer.feature_names


def test_explain_returns_top_n_features(pipeline, sample_row):
    explainer = SessionExplainer(pipeline)
    result = explainer.explain(sample_row, top_n=5)
    assert len(result) == 5
    for item in result:
        assert set(item.keys()) == {"feature", "shap_value", "direction"}
        assert item["direction"] in {"increases", "decreases"}


def test_explain_sorted_by_absolute_influence(pipeline, sample_row):
    explainer = SessionExplainer(pipeline)
    result = explainer.explain(sample_row, top_n=10)
    magnitudes = [abs(item["shap_value"]) for item in result]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_rejects_multi_row_input(pipeline, settings):
    df = engineer_features(load_sessions(settings=settings))
    X = df.drop(columns=[settings.data.target_column])
    explainer = SessionExplainer(pipeline)
    with pytest.raises(ValueError):
        explainer.explain(X.iloc[:2], top_n=5)
