"""Tests for src/features/engineering.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.features.engineering import ENGINEERED_FEATURE_COLUMNS, engineer_features


@pytest.fixture(scope="module")
def df():
    return load_sessions(settings=get_settings())


def test_engineer_features_adds_expected_columns(df):
    out = engineer_features(df)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns


def test_engineer_features_no_nan_or_inf(df):
    out = engineer_features(df)
    numeric_new = out[ENGINEERED_FEATURE_COLUMNS].select_dtypes(include=[np.number])
    assert not numeric_new.isna().any().any()
    assert np.isfinite(numeric_new).all().all()


def test_engineer_features_does_not_mutate_input(df):
    original_cols = set(df.columns)
    engineer_features(df)
    assert set(df.columns) == original_cols


def test_product_page_ratio_is_bounded(df):
    out = engineer_features(df)
    assert (out["product_page_ratio"] >= 0).all()
    assert (out["product_page_ratio"] <= 1).all()


def test_is_returning_visitor_is_binary(df):
    out = engineer_features(df)
    assert set(out["is_returning_visitor"].unique()) <= {0, 1}


def test_missing_required_column_raises_keyerror(df):
    broken = df.drop(columns=["VisitorType"])
    with pytest.raises(KeyError):
        engineer_features(broken)
