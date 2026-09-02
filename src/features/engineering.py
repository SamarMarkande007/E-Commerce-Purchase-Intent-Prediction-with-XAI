"""Behavioural feature engineering.

The five features here were prototyped and justified in
``notebooks/02_features.ipynb`` (see that notebook for the EDA-driven
rationale and the feature-importance analysis behind each one). This
module is the single reusable implementation — preprocessing, the API,
and the dashboard all call :func:`engineer_features` instead of
reimplementing the logic, so a session is guaranteed to be transformed
identically everywhere.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Names of the columns this module adds, in the order they're created.
#: Exposed so preprocessing/config can reference the full feature list
#: without hard-coding the names in two places.
ENGINEERED_FEATURE_COLUMNS: list[str] = [
    "total_pages_viewed",
    "total_duration",
    "product_page_ratio",
    "avg_time_per_product_page",
    "is_returning_visitor",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural ratio/aggregate features to a sessions DataFrame.

    Adds five columns (see :data:`ENGINEERED_FEATURE_COLUMNS`):

    - ``total_pages_viewed``: Administrative + Informational + ProductRelated.
    - ``total_duration``: sum of the three ``*_Duration`` columns.
    - ``product_page_ratio``: share of pages viewed that were product
      pages; 0 for sessions with zero pages viewed (avoids divide-by-zero).
    - ``avg_time_per_product_page``: time spent per product page; 0 for
      sessions with zero product pages viewed.
    - ``is_returning_visitor``: 1 if ``VisitorType == "Returning_Visitor"``,
      else 0.

    Args:
        df: A sessions DataFrame containing at least the columns
            ``Administrative``, ``Informational``, ``ProductRelated``,
            their ``*_Duration`` counterparts, and ``VisitorType``.

    Returns:
        A copy of ``df`` with the five engineered columns appended.
        Does not mutate the input DataFrame.

    Raises:
        KeyError: If a required source column is missing.
    """
    required = {
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "VisitorType",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"engineer_features is missing required column(s): {sorted(missing)}")

    out = df.copy()

    out["total_pages_viewed"] = (
        out["Administrative"] + out["Informational"] + out["ProductRelated"]
    )
    out["total_duration"] = (
        out["Administrative_Duration"]
        + out["Informational_Duration"]
        + out["ProductRelated_Duration"]
    )
    out["product_page_ratio"] = np.where(
        out["total_pages_viewed"] > 0,
        out["ProductRelated"] / out["total_pages_viewed"],
        0.0,
    )
    out["avg_time_per_product_page"] = np.where(
        out["ProductRelated"] > 0,
        out["ProductRelated_Duration"] / out["ProductRelated"],
        0.0,
    )
    out["is_returning_visitor"] = (out["VisitorType"] == "Returning_Visitor").astype(int)

    logger.debug("Engineered %d new feature columns for %d rows", len(ENGINEERED_FEATURE_COLUMNS), len(out))

    return out
