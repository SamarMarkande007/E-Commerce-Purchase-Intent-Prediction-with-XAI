"""Tests for src/data/loader.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.utils.exceptions import DataValidationError


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_load_sessions_returns_nonempty_dataframe(settings):
    df = load_sessions(settings=settings)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_sessions_has_all_expected_columns(settings):
    df = load_sessions(settings=settings)
    expected = set(settings.all_feature_columns) | {settings.data.target_column}
    assert expected.issubset(set(df.columns))


def test_categorical_columns_are_cast_correctly(settings):
    df = load_sessions(settings=settings)
    for col in settings.data.categorical_columns:
        assert str(df[col].dtype) == "category", f"{col} was not cast to category dtype"


def test_int_coded_traps_are_categorical_not_numeric(settings):
    """The four columns most likely to be misused (Region, Browser, etc.)
    must not be treated as continuous numbers."""
    df = load_sessions(settings=settings)
    for col in settings.data.int_coded_categorical_columns:
        assert str(df[col].dtype) == "category", (
            f"{col} is int-coded but must be categorical, not numeric"
        )


def test_target_is_binary_with_no_nulls(settings):
    df = load_sessions(settings=settings)
    target = df[settings.data.target_column]
    assert target.isna().sum() == 0
    assert set(target.unique()) <= {0, 1}


def test_missing_file_raises_data_validation_error(settings):
    with pytest.raises(DataValidationError):
        load_sessions(csv_path="does/not/exist.csv", settings=settings)
