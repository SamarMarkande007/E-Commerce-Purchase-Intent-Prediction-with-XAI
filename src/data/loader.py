"""Data ingestion and validation.

This is the single place the raw dataset is read from disk. Every
notebook, the API, and the dashboard should call :func:`load_sessions`
instead of calling ``pd.read_csv`` directly — that's what guarantees
everyone sees the same dtypes (in particular, the four int-coded
categorical columns) and the same validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def _validate_schema(df: pd.DataFrame, settings: Settings) -> None:
    """Check that every expected column is present.

    Raises:
        DataValidationError: If any required column (features + target)
            is missing from the DataFrame.
    """
    expected_columns = set(settings.all_feature_columns) | {settings.data.target_column}
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns

    if missing:
        raise DataValidationError(
            f"Dataset is missing expected column(s): {sorted(missing)}. "
            f"Check src/config/config.yaml against the CSV header."
        )


def _validate_target(df: pd.DataFrame, settings: Settings) -> None:
    """Check the target column is binary (0/1) with no missing values.

    Raises:
        DataValidationError: If the target has nulls or values other than
            0/1.
    """
    target = df[settings.data.target_column]

    if target.isna().any():
        n_missing = int(target.isna().sum())
        raise DataValidationError(
            f"Target column '{settings.data.target_column}' has "
            f"{n_missing} missing value(s); cannot train on an "
            f"incomplete target."
        )

    unexpected = set(target.unique()) - {0, 1}
    if unexpected:
        raise DataValidationError(
            f"Target column '{settings.data.target_column}' contains "
            f"non-binary value(s): {sorted(unexpected)}. Expected only 0/1."
        )


def _cast_categorical_dtypes(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Cast declared categorical columns (including the int-coded ones) to
    pandas 'category' dtype.

    This is the fix for the dataset's main trap: OperatingSystems,
    Browser, Region, and TrafficType are stored as integers but are IDs,
    not quantities. Left as int64, a scaler or a model would treat
    "Region 3" as three times "Region 1," which is meaningless. Casting
    to 'category' here means every downstream step (EDA, preprocessing,
    the API schema) sees these correctly as categories by default.
    """
    df = df.copy()
    for col in settings.data.categorical_columns:
        df[col] = df[col].astype("category")
    return df


def load_sessions(
    csv_path: str | Path | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Load and validate the e-commerce sessions dataset.

    Args:
        csv_path: Optional override for the CSV location. Defaults to
            the path in config.yaml.
        settings: Optional pre-loaded :class:`Settings` object, mainly
            for tests. Defaults to the cached global settings.

    Returns:
        A validated DataFrame with categorical columns (including the
        int-coded ID columns) cast to pandas 'category' dtype.

    Raises:
        DataValidationError: If the file is missing, empty, unreadable,
            missing expected columns, or has an invalid target column.
    """
    settings = settings or get_settings()
    path = Path(csv_path) if csv_path is not None else settings.paths.dataset_csv

    if not path.exists():
        raise DataValidationError(
            f"Dataset CSV not found at {path}. "
            f"Check src/config/config.yaml 'paths.dataset_csv'."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset CSV at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Dataset CSV at {path} could not be parsed: {exc}") from exc

    if df.empty:
        raise DataValidationError(f"Dataset CSV at {path} loaded but contains 0 rows.")

    _validate_schema(df, settings)
    _validate_target(df, settings)
    df = _cast_categorical_dtypes(df, settings)

    logger.info(
        "Loaded %d sessions (%d columns) from %s. Conversion rate: %.2f%%",
        len(df),
        df.shape[1],
        path,
        100 * df[settings.data.target_column].mean(),
    )

    return df
