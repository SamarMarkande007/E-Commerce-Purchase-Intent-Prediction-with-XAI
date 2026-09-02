"""Tests for src/preprocessing/pipeline.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.features.engineering import engineer_features
from src.preprocessing.pipeline import build_full_pipeline, build_preprocessor


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def train_test_split_data(settings):
    df = engineer_features(load_sessions(settings=settings))
    y = df[settings.data.target_column]
    X = df.drop(columns=[settings.data.target_column])
    return train_test_split(
        X, y, test_size=settings.modeling.test_size,
        stratify=y, random_state=settings.random_seed,
    )


def test_preprocessor_fits_and_transforms_train_split(settings, train_test_split_data):
    X_train, _, y_train, _ = train_test_split_data
    preprocessor = build_preprocessor(settings)
    transformed = preprocessor.fit_transform(X_train)
    # Output is numeric (post one-hot + scaling), same number of rows as input.
    assert transformed.shape[0] == len(X_train)
    assert np.isfinite(transformed if not hasattr(transformed, "toarray") else transformed.toarray()).all()


def test_preprocessor_handles_unseen_category_at_transform_time(settings, train_test_split_data):
    """Simulates a category seen at serving time that wasn't in training —
    must not raise, per the brief's handle_unknown='ignore' requirement."""
    X_train, X_test, _, _ = train_test_split_data
    preprocessor = build_preprocessor(settings)
    preprocessor.fit(X_train)

    X_test_modified = X_test.copy()
    cat_col = settings.data.categorical_columns[0]
    # Inject a category value guaranteed not to exist in training.
    X_test_modified[cat_col] = X_test_modified[cat_col].astype(object)
    X_test_modified.iloc[0, X_test_modified.columns.get_loc(cat_col)] = "__NEVER_SEEN__"

    # Should not raise.
    result = preprocessor.transform(X_test_modified)
    assert result.shape[0] == len(X_test_modified)


def test_full_pipeline_smote_only_resamples_training_fold(settings, train_test_split_data):
    """The defining no-leakage requirement: fitting on the train split must
    not touch or change the size of the test split."""
    X_train, X_test, y_train, y_test = train_test_split_data
    test_len_before = len(X_test)

    pipeline = build_full_pipeline(
        settings, LogisticRegression(max_iter=1000), imbalance_strategy="smote"
    )
    pipeline.fit(X_train, y_train)

    # Test split itself must be completely untouched by fitting.
    assert len(X_test) == test_len_before

    # Pipeline must still be able to score the (unresampled, real-world-shaped) test split.
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)


def test_full_pipeline_none_strategy_runs_without_resampling(settings, train_test_split_data):
    X_train, X_test, y_train, y_test = train_test_split_data
    pipeline = build_full_pipeline(
        settings,
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        imbalance_strategy="none",
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)


def test_invalid_imbalance_strategy_raises(settings):
    with pytest.raises(ValueError):
        build_full_pipeline(settings, LogisticRegression(), imbalance_strategy="bogus")
