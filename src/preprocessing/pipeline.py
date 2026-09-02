"""Preprocessing: encoding, scaling, and the class-imbalance strategy.

Everything here is built as an *unfitted* pipeline. It is fit only
inside cross-validation folds (Phase 6) or once on the training split
before saving for serving — never on the full dataset before splitting.
Fitting a OneHotEncoder or SMOTE on all the data leaks information about
rows the model should never have seen during evaluation and inflates
every reported metric. See notebooks/03_modeling.ipynb for that split.

Because the imbalance strategy (SMOTE) must resample only the training
fold, this uses imbalanced-learn's ``Pipeline`` rather than sklearn's —
sklearn's ``Pipeline`` cannot contain a resampling step (a step that
changes the number of rows), imblearn's can.
"""

from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from src.config.settings import Settings
from src.features.engineering import ENGINEERED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

#: Imbalance strategies benchmarked in Phase 6 (notebooks/03_modeling.ipynb).
#: "none" is included as a baseline so the benefit of SMOTE/class_weight can
#: be measured against doing nothing, per the brief's "benchmark, don't
#: assume" hint.
ImbalanceStrategy = str  # one of: "smote", "none"  (class_weight is set on the estimator, not here)


def build_preprocessor(settings: Settings) -> ColumnTransformer:
    """Build the (unfitted) encoding + scaling ColumnTransformer.

    - Duration columns get ``log1p`` (they are heavily right-skewed with
      many zeros) followed by ``RobustScaler``.
    - Other numeric columns get ``RobustScaler`` only.
    - Categorical columns (including the int-coded ID traps —
      OperatingSystems, Browser, Region, TrafficType) are one-hot encoded
      with ``handle_unknown="ignore"`` so a category unseen during
      training (e.g. a new Region ID in production) does not crash the
      API — it's encoded as all-zeros instead of raising.

    Args:
        settings: Project settings; supplies the column lists so this
            function never hard-codes a column name.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`. Call
        ``.fit_transform`` on a *training* split only.
    """
    duration_cols = settings.data.duration_columns
    other_numeric_cols = [
        c for c in settings.data.numeric_columns if c not in duration_cols
    ] + [
        c for c in ENGINEERED_FEATURE_COLUMNS
        if c not in ("is_returning_visitor",)  # binary flag doesn't need scaling
    ]
    categorical_cols = settings.data.categorical_columns
    passthrough_cols = ["is_returning_visitor"]  # already 0/1, no transform needed

    duration_pipeline = ImbPipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("duration", duration_pipeline, duration_cols),
            ("numeric", RobustScaler(), other_numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("passthrough", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )

    logger.debug(
        "Built preprocessor: %d duration cols, %d other numeric cols, "
        "%d categorical cols, %d passthrough cols",
        len(duration_cols), len(other_numeric_cols), len(categorical_cols), len(passthrough_cols),
    )

    return preprocessor


def build_full_pipeline(
    settings: Settings,
    estimator: BaseEstimator | ClassifierMixin,
    imbalance_strategy: ImbalanceStrategy = "none",
) -> ImbPipeline:
    """Build the complete (unfitted) preprocessing + imbalance + model pipeline.

    Wraps :func:`build_preprocessor` together with an optional SMOTE step
    and the given estimator in a single imblearn ``Pipeline``, so that
    ``pipeline.fit(X_train, y_train)`` fits encoding, scaling, resampling,
    and the model together, correctly scoped to only the training fold.

    Args:
        settings: Project settings.
        estimator: An unfitted scikit-learn-compatible classifier
            (e.g. LogisticRegression(), RandomForestClassifier()).
        imbalance_strategy: ``"smote"`` to oversample the minority class
            inside the pipeline, or ``"none"`` to skip resampling (e.g.
            when the estimator instead uses ``class_weight="balanced"``
            or ``scale_pos_weight``, set by the caller on ``estimator``
            itself). Benchmarking both is done in Phase 6.

    Returns:
        An unfitted imblearn :class:`Pipeline` combining preprocessing,
        (optional) resampling, and the estimator.

    Raises:
        ValueError: If ``imbalance_strategy`` is not a recognised value.
    """
    if imbalance_strategy not in ("smote", "none"):
        raise ValueError(
            f"Unknown imbalance_strategy: {imbalance_strategy!r}. Expected 'smote' or 'none'."
        )

    preprocessor = build_preprocessor(settings)
    steps = [("preprocessor", preprocessor)]

    if imbalance_strategy == "smote":
        steps.append(("smote", SMOTE(random_state=settings.random_seed)))

    steps.append(("classifier", estimator))

    pipeline = ImbPipeline(steps=steps)
    logger.debug(
        "Built full pipeline with imbalance_strategy=%s, estimator=%s",
        imbalance_strategy, type(estimator).__name__,
    )
    return pipeline
