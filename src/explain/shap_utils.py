"""SHAP-based per-session explanations, for real-time use by the API.

Extracted from notebooks/04_xai.ipynb — same TreeExplainer, same
log-odds interpretation, same feature-name cleanup, now wrapped for
repeated calls against a single session rather than a notebook-style
batch analysis. See the notebook for the global-importance analysis
and the reasoning behind treating PageValues as the dominant feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)


class SessionExplainer:
    """Wraps a fitted pipeline's classifier with a SHAP TreeExplainer.

    Built once at API startup (constructing the explainer is not free)
    and reused for every /predict request.
    """

    def __init__(self, pipeline: ImbPipeline):
        """Build the explainer from a fitted preprocessing+model pipeline.

        Args:
            pipeline: A fitted pipeline with ``"preprocessor"`` and
                ``"classifier"`` named steps, as produced by
                ``src.preprocessing.pipeline.build_full_pipeline`` and
                saved by ``src.models.trainer.ModelTrainer.save``.

        Raises:
            KeyError: If the pipeline is missing the expected named steps.
        """
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(self.classifier)

        raw_names = self.preprocessor.get_feature_names_out()
        self.feature_names = [n.split("__", 1)[1] if "__" in n else n for n in raw_names]

        logger.info(
            "SessionExplainer ready: %s, %d transformed features",
            type(self.classifier).__name__, len(self.feature_names),
        )

    def explain(self, X_row: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Explain a single session's prediction.

        Args:
            X_row: A single-row DataFrame of raw (pre-preprocessing)
                feature values — same shape the pipeline's
                ``.predict_proba`` expects.
            top_n: How many of the most influential features to return,
                ranked by absolute SHAP value.

        Returns:
            A list of ``top_n`` dicts, each with:

            - ``feature``: cleaned feature name (e.g. ``"PageValues"``).
            - ``shap_value``: the raw SHAP contribution, on the model's
              log-odds scale — positive pushes toward "purchase,"
              negative pushes toward "no purchase."
            - ``direction``: ``"increases"`` or ``"decreases"``, a
              human-readable version of the sign, for display.

            Sorted by absolute SHAP value, largest influence first.

        Raises:
            ValueError: If ``X_row`` does not contain exactly one row.
        """
        if len(X_row) != 1:
            raise ValueError(f"explain() expects exactly one row, got {len(X_row)}")

        X_transformed = self.preprocessor.transform(X_row)
        X_dense = np.asarray(X_transformed.todense()) if hasattr(X_transformed, "todense") else np.asarray(X_transformed)

        shap_values = self._explainer.shap_values(X_dense)[0]

        top_idx = np.argsort(-np.abs(shap_values))[:top_n]
        return [
            {
                "feature": self.feature_names[i],
                "shap_value": float(shap_values[i]),
                "direction": "increases" if shap_values[i] > 0 else "decreases",
            }
            for i in top_idx
        ]
