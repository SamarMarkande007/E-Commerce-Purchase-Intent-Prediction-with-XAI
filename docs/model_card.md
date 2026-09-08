# Model Card — Purchase-Intent Prediction

## Intended use

Predicts the probability that an active e-commerce session will end in a
purchase, using only within-session behavioural signals (page views,
durations, bounce/exit rates, calendar context, visitor type). Intended
to support session-level interventions (e.g. a checkout-assist prompt or
a targeted discount) — **not** intended for long-term customer scoring,
credit-like decisions, or any use outside real-time session analysis.

## Model

CatBoost classifier (`class_weight`-balanced), selected from a 6-model,
2-imbalance-strategy comparison (see `notebooks/03_modeling.ipynb`).
Wrapped in a scikit-learn/imblearn pipeline that also performs all
encoding and scaling, so a raw session can be scored end-to-end via
`models/best_pipeline.joblib`.

## Performance

*(Held-out test set — `notebooks/03_modeling.ipynb`)*

| Metric | Value |
|---|---|
| PR-AUC | 0.854 |
| ROC-AUC | 0.971 |
| Recall at chosen threshold (~0.014) | ≈ 99.7% |
| Precision at chosen threshold | ≈ 43.3% |

The decision threshold was chosen to maximise expected business value
under assumed figures of a $100 captured conversion vs. a $5
intervention cost (see `notebooks/03_modeling.ipynb` ) — **these
threshold-dependent numbers will change substantially if the real
business assumptions differ**. Replace the placeholder figures before
trusting this threshold in production.

## Key limitation — `PageValues` dependency

This is the most important caveat for this model. `PageValues` accounts
for the majority of the model's explanatory power:

- SHAP global importance ranks it far above every other feature (see
  `notebooks/04_xai.ipynb` ).
- The Phase 6 with/without ablation showed a **35% relative PR-AUC
  drop** when it is removed (0.854 → 0.553).
- SHAP local explanations show it as the single largest contributor,
  in both directions, for individual sessions (`notebooks/04_xai.ipynb`
  ), independently confirmed by LIME.

**This model assumes `PageValues` is genuinely known at the moment of
scoring** - e.g. late in the session, at checkout initiation. If the
actual deployment scores sessions earlier, before `PageValues` for the
current session is meaningfully populated, this model's performance
claims do not hold. Use `models/best_pipeline_no_pagevalues.joblib`
(PR-AUC 0.553) instead in that case, and expect materially weaker
performance as the honest cost of a feature that isn't actually
available at prediction time.

## Other limitations

- Trained on 12,000 sessions from one dataset; behaviour on a different
  site, traffic mix, or season is untested.
- The ~15.7% class imbalance was handled with
  `class_weight="balanced"`; a large shift in real-world conversion
  rate (e.g. a flash sale) may require retraining or re-tuning the
  threshold, not just re-scoring with the existing model.
- Explanations (SHAP/LIME) describe correlational behavioural patterns
  the model learned, not causal drivers of purchase intent — "high
  `PageValues` predicts purchase" is not the same claim as "showing a
  user a high-`PageValues` page causes them to buy."
- The two placeholder business values ($100 conversion, $5
  intervention) used for threshold selection are illustrative, not
  researched, and should be replaced with real figures before
  production use.

## Explainability methods used

- **SHAP** (`TreeExplainer`) — global feature importance and local,
  per-session explanations, operating on the model's log-odds
  (margin) scale.
- **LIME** (`LimeTabularExplainer`) — an independent, locally-linear
  approximation for individual sessions, used as a cross-check against
  SHAP rather than a replacement for it.

Both methods agreed on the dominant feature and its direction for the
two example sessions examined in `notebooks/04_xai.ipynb` — genuine
independent confirmation, since the two methods work in structurally
different ways (tree-based exact attribution vs. local surrogate
model).

## Provenance

- Training notebook: `notebooks/03_modeling.ipynb`
- Explainability notebook: `notebooks/04_xai.ipynb`
- Random seed: 42 (fixed throughout, per `src/config/config.yaml`)
- MLflow registered model: `purchase-intent-catboost`, version 1
