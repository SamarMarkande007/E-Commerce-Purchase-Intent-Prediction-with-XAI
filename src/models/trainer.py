"""Model training: six algorithms behind one common interface.

Wraps the model-comparison logic prototyped in
notebooks/03_modeling.ipynb into a reusable ``ModelTrainer`` class, so
adding a 7th algorithm later means adding one dict entry, not
duplicating a training loop. The API (Phase 8) does not use this class
directly — it loads the already-trained pipeline saved by the
notebook — but a future retraining script would.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.settings import Settings
from src.models.evaluation import SKLEARN_SCORING
from src.preprocessing.pipeline import ImbalanceStrategy, build_full_pipeline

logger = logging.getLogger(__name__)

#: The six algorithms compared in notebooks/03_modeling.ipynb, per the brief's requirement.
CANDIDATE_MODEL_NAMES: list[str] = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]


class ModelTrainer:
    """Trains and compares the six candidate algorithms with a common interface.

    Usage::

        trainer = ModelTrainer(settings)
        results_df = trainer.compare_models(X_train, y_train)
        best_name, best_strategy = trainer.select_best(results_df)
        pipeline = trainer.fit_best(X_train, y_train, best_name, best_strategy)
        trainer.save(pipeline, "models/best_pipeline.joblib")
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._scale_pos_weight: float | None = None

    def get_candidate_models(
        self, balanced: bool, scale_pos_weight: float = 1.0
    ) -> dict[str, ClassifierMixin]:
        """Return the six candidate estimators.

        Args:
            balanced: If True, each estimator uses its native imbalance
                handling (``class_weight="balanced"`` for most models,
                ``scale_pos_weight`` for the two boosting models that
                use that parameter name instead). If False, returns
                plain unweighted estimators (used with the SMOTE
                strategy instead, so the two approaches aren't stacked).
            scale_pos_weight: Ratio of negative to positive class counts
                in the training data, used only when ``balanced=True``.
                Ignored otherwise.

        Returns:
            A dict mapping model name (from :data:`CANDIDATE_MODEL_NAMES`)
            to an unfitted, scikit-learn-compatible classifier instance.
        """
        seed = self.settings.random_seed
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=seed,
                class_weight="balanced" if balanced else None,
            ),
            "decision_tree": DecisionTreeClassifier(
                random_state=seed,
                max_depth=10,
                class_weight="balanced" if balanced else None,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                max_depth=12,
                class_weight="balanced" if balanced else None,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                random_state=seed,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight if balanced else 1.0,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=300,
                random_state=seed,
                verbosity=-1,
                class_weight="balanced" if balanced else None,
            ),
            "catboost": CatBoostClassifier(
                iterations=300,
                random_state=seed,
                verbose=0,
                auto_class_weights="Balanced" if balanced else None,
            ),
        }

    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategies: tuple[str, ...] = ("class_weight", "smote"),
    ) -> pd.DataFrame:
        """Cross-validate all six models under each imbalance strategy.

        Args:
            X_train: Training features (pre-preprocessing; the pipeline
                built internally handles encoding/scaling/resampling).
            y_train: Training target.
            strategies: Which imbalance strategies to compare. Each
                element must be ``"class_weight"`` or ``"smote"``.

        Returns:
            A DataFrame with one row per (model, strategy) combination,
            columns for ``model``, ``strategy``, and each metric in
            :data:`~src.models.evaluation.SKLEARN_SCORING`, sorted by
            PR-AUC descending (best first).
        """
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        cv = StratifiedKFold(
            n_splits=self.settings.modeling.cv_folds,
            shuffle=True,
            random_state=self.settings.random_seed,
        )

        results = []
        for strategy in strategies:
            balanced = strategy == "class_weight"
            models = self.get_candidate_models(balanced=balanced, scale_pos_weight=scale_pos_weight)

            for name, estimator in models.items():
                pipeline = build_full_pipeline(
                    self.settings,
                    estimator,
                    imbalance_strategy="smote" if strategy == "smote" else "none",
                )
                scores = cross_validate(
                    pipeline, X_train, y_train, cv=cv, scoring=SKLEARN_SCORING, n_jobs=1
                )

                row: dict[str, Any] = {"model": name, "strategy": strategy}
                for metric in SKLEARN_SCORING:
                    row[metric] = scores[f"test_{metric}"].mean()
                results.append(row)

                logger.info(
                    "%s / %s -> PR-AUC=%.3f, recall=%.3f",
                    name,
                    strategy,
                    row["pr_auc"],
                    row["recall"],
                )

        return pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    def select_best(self, results_df: pd.DataFrame, metric: str = "pr_auc") -> tuple[str, str]:
        """Pick the top (model, strategy) combination by a given metric.

        Args:
            results_df: Output of :meth:`compare_models`.
            metric: Column to rank by. Defaults to PR-AUC, per the
                brief's requirement that it (or recall) lead over
                accuracy on this imbalanced target.

        Returns:
            A tuple ``(model_name, strategy)`` for the top row.

        Raises:
            ValueError: If ``results_df`` is empty.
        """
        if results_df.empty:
            raise ValueError("results_df is empty; nothing to select from")

        best_row = results_df.sort_values(metric, ascending=False).iloc[0]
        return best_row["model"], best_row["strategy"]

    def fit_best(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        strategy: str,
    ) -> ImbPipeline:
        """Build and fit the full pipeline for a chosen (model, strategy) pair.

        Args:
            X_train: Training features.
            y_train: Training target.
            model_name: One of :data:`CANDIDATE_MODEL_NAMES`.
            strategy: ``"class_weight"`` or ``"smote"``.

        Returns:
            The fitted imblearn pipeline (preprocessing + optional SMOTE
            + classifier), ready for ``.predict_proba()``.

        Raises:
            KeyError: If ``model_name`` is not a recognised candidate.
        """
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        balanced = strategy == "class_weight"
        models = self.get_candidate_models(balanced=balanced, scale_pos_weight=scale_pos_weight)
        if model_name not in models:
            raise KeyError(
                f"Unknown model_name: {model_name!r}. Expected one of {CANDIDATE_MODEL_NAMES}"
            )

        imbalance_strategy: ImbalanceStrategy = "smote" if strategy == "smote" else "none"
        pipeline = build_full_pipeline(
            self.settings, models[model_name], imbalance_strategy=imbalance_strategy
        )
        pipeline.fit(X_train, y_train)
        return pipeline

    @staticmethod
    def save(pipeline: ImbPipeline, path: str | Path) -> None:
        """Save a fitted pipeline to disk with joblib.

        Args:
            pipeline: A fitted pipeline (from :meth:`fit_best`).
            path: Destination file path; parent directories are created
                if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path)
        logger.info("Saved pipeline to %s", path)

    @staticmethod
    def load(path: str | Path) -> ImbPipeline:
        """Load a previously saved pipeline.

        Args:
            path: Path to a .joblib file saved by :meth:`save`.

        Returns:
            The fitted pipeline, ready for ``.predict_proba()``.
        """
        return joblib.load(Path(path))
