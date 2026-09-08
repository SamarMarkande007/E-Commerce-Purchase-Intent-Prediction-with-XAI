"""Purchase-Intent XAI — Streamlit Dashboard.

Six sections (per the brief's §4.8): overview, live session scoring,
funnel/cohort analytics, model comparison, explainability, performance
metrics. This dashboard NEVER retrains a model — it reads the artifacts
saved by notebooks/03_modeling.ipynb and notebooks/04_xai.ipynb, and for
live scoring, calls the FastAPI service (src/api/main.py) so there is
exactly one source of truth for how a session gets scored.

Run with:  streamlit run dashboard/app.py
Requires the API running separately for the live-scoring section:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.model_selection import train_test_split

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.settings import get_settings
from src.data.loader import load_sessions
from src.explain.shap_utils import SessionExplainer
from src.features.engineering import ENGINEERED_FEATURE_COLUMNS, engineer_features
from src.utils.logging_config import setup_logging

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Purchase-Intent XAI", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit caching per the brief's requirement. Data and
# small artifacts use cache_data (safe to hash/copy); heavy model objects
# use cache_resource (loaded once, shared, never copied).
# ---------------------------------------------------------------------------

@st.cache_resource
def get_settings_cached():
    settings = get_settings()
    setup_logging(level="INFO", log_file=settings.paths.log_file)
    return settings


@st.cache_data
def get_engineered_data():
    settings = get_settings_cached()
    df = engineer_features(load_sessions(settings=settings))
    return df


@st.cache_resource
def get_pipeline():
    settings = get_settings_cached()
    path = settings.paths.models_dir / "best_pipeline.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def get_explainer():
    pipeline = get_pipeline()
    if pipeline is None:
        return None
    return SessionExplainer(pipeline)


@st.cache_data
def get_model_metadata():
    settings = get_settings_cached()
    path = settings.paths.models_dir / "best_model_metadata.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_comparison_results():
    settings = get_settings_cached()
    path = settings.paths.models_dir / "model_comparison_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def get_test_split():
    """Reconstruct the same held-out test split used in training, purely
    for reporting saved-model performance — no .fit() call happens here,
    only .predict_proba() on the already-trained pipeline."""
    settings = get_settings_cached()
    df = get_engineered_data()
    target = settings.data.target_column
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=settings.modeling.test_size,
        stratify=y, random_state=settings.random_seed,
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_overview():
    st.title("Purchase-Intent XAI")
    st.caption("Predicting and explaining e-commerce purchase intent, end to end.")

    df = get_engineered_data()
    settings = get_settings_cached()
    target = settings.data.target_column

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", f"{len(df):,}")
    col2.metric("Conversion rate", f"{df[target].mean():.1%}")
    col3.metric("Converted sessions", f"{df[target].sum():,}")
    col4.metric("Non-converted sessions", f"{(df[target] == 0).sum():,}")

    st.markdown("""
This project predicts whether an active e-commerce session will end in a
purchase, using only within-session behavioural signals — no personal
data, no cross-session history. Full methodology in `notebooks/`, model
reasoning in `docs/model_card.md`.
""")

    class_counts = df[target].value_counts().rename({0: "No purchase", 1: "Purchase"})
    fig = px.bar(
        x=class_counts.index, y=class_counts.values,
        labels={"x": "Outcome", "y": "Sessions"},
        title="Class balance — the central modelling challenge",
        color=class_counts.index,
        color_discrete_map={"No purchase": "#888780", "Purchase": "#1D9E75"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.info(
        "Only ~15.7% of sessions convert. Every model and metric choice in this "
        "project accounts for this imbalance — see the 'Model comparison' section."
    )


def section_live_scoring():
    st.title("Live session scoring")
    st.caption("Calls the FastAPI service — this dashboard never scores sessions itself.")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        health.raise_for_status()
        health_data = health.json()
        st.success(f"API connected — model: {health_data['model_name']}, threshold: {health_data['decision_threshold']:.4f}")
    except requests.exceptions.RequestException:
        st.error(
            f"Cannot reach the API at {API_URL}. Start it in a separate terminal with:\n\n"
            f"`uvicorn src.api.main:app --reload`"
        )
        return

    with st.form("scoring_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Page activity")
            administrative = st.number_input("Administrative pages", min_value=0, value=2)
            administrative_duration = st.number_input("Administrative duration (s)", min_value=0.0, value=40.0)
            informational = st.number_input("Informational pages", min_value=0, value=0)
            informational_duration = st.number_input("Informational duration (s)", min_value=0.0, value=0.0)
            product_related = st.number_input("Product-related pages", min_value=0, value=25)
            product_related_duration = st.number_input("Product-related duration (s)", min_value=0.0, value=620.0)

        with col2:
            st.subheader("Engagement signals")
            bounce_rates = st.slider("Bounce rate", 0.0, 1.0, 0.01)
            exit_rates = st.slider("Exit rate", 0.0, 1.0, 0.02)
            page_values = st.number_input(
                "PageValues", min_value=0.0, value=15.0,
                help="Dominant feature — see docs/model_card.md. Assumes this is known at scoring time.",
            )
            special_day = st.slider("Special-day closeness", 0.0, 1.0, 0.0)

        with col3:
            st.subheader("Context")
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=10)
            visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
            weekend = st.checkbox("Weekend session", value=False)
            operating_systems = st.selectbox("OS ID (anonymised)", list(range(1, 9)), index=1)
            browser = st.selectbox("Browser ID (anonymised)", list(range(1, 14)), index=1)
            region = st.selectbox("Region ID (anonymised)", list(range(1, 10)), index=0)
            traffic_type = st.selectbox("Traffic type ID (anonymised)", list(range(1, 21)), index=1)

        submitted = st.form_submit_button("Score this session")

    if submitted:
        payload = {
            "Administrative": administrative, "Administrative_Duration": administrative_duration,
            "Informational": informational, "Informational_Duration": informational_duration,
            "ProductRelated": product_related, "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates, "ExitRates": exit_rates, "PageValues": page_values,
            "SpecialDay": special_day, "Month": month, "OperatingSystems": operating_systems,
            "Browser": browser, "Region": region, "TrafficType": traffic_type,
            "VisitorType": visitor_type, "Weekend": weekend,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
            return

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Purchase probability", f"{result['purchase_probability']:.1%}")
        col2.metric("Decision", "Will purchase" if result["will_purchase"] else "Won't purchase")
        col3.metric("Confidence", result["confidence"].capitalize())

        factors_df = pd.DataFrame(result["top_factors"])
        factors_df["color"] = factors_df["direction"].map({"increases": "#1D9E75", "decreases": "#D85A30"})
        fig = go.Figure(go.Bar(
            x=factors_df["shap_value"], y=factors_df["feature"], orientation="h",
            marker_color=factors_df["color"],
        ))
        fig.update_layout(
            title="Top contributing factors (SHAP, log-odds scale)",
            xaxis_title="Contribution (positive = toward purchase)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width='stretch')


def section_funnel_cohort():
    st.title("Funnel & cohort analytics")
    df = get_engineered_data()
    settings = get_settings_cached()
    target = settings.data.target_column

    breakdown_options = {
        "Month": "Month", "Visitor type": "VisitorType",
        "Weekend vs weekday": "Weekend", "Region": "Region",
    }
    choice = st.selectbox("Break down conversion rate by:", list(breakdown_options.keys()))
    col = breakdown_options[choice]

    grouped = df.groupby(col, observed=True)[target].agg(["mean", "count"]).reset_index()
    grouped.columns = [col, "conversion_rate", "sessions"]
    grouped = grouped.sort_values("conversion_rate", ascending=False)

    fig = px.bar(
        grouped, x=col, y="conversion_rate", color="conversion_rate",
        hover_data=["sessions"], color_continuous_scale="Greens",
        title=f"Conversion rate by {choice}",
        labels={"conversion_rate": "Conversion rate"},
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Engagement funnel")
    funnel_cols = ["Administrative", "Informational", "ProductRelated"]
    funnel_df = pd.DataFrame({
        "Stage": funnel_cols,
        "Avg. pages viewed (converted)": [df.loc[df[target] == 1, c].mean() for c in funnel_cols],
        "Avg. pages viewed (not converted)": [df.loc[df[target] == 0, c].mean() for c in funnel_cols],
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Converted", x=funnel_df["Stage"], y=funnel_df["Avg. pages viewed (converted)"], marker_color="#1D9E75"))
    fig2.add_trace(go.Bar(name="Not converted", x=funnel_df["Stage"], y=funnel_df["Avg. pages viewed (not converted)"], marker_color="#888780"))
    fig2.update_layout(barmode="group", title="Average pages viewed by stage, converted vs. not")
    st.plotly_chart(fig2, width='stretch')

    st.subheader("Cohort heatmap — Visitor type × Month")
    cohort = df.pivot_table(index="VisitorType", columns="Month", values=target, aggfunc="mean", observed=True)
    month_order = [m for m in ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if m in cohort.columns]
    cohort = cohort[month_order]
    fig3 = px.imshow(
        cohort, color_continuous_scale="Greens", aspect="auto",
        labels=dict(color="Conversion rate"),
        title="Conversion rate: Visitor type × Month",
    )
    st.plotly_chart(fig3, width='stretch')


def section_model_comparison():
    st.title("Model comparison")
    st.caption("Read from notebooks/03_modeling.ipynb's saved comparison — never recomputed here.")

    results_df = get_comparison_results()
    metadata = get_model_metadata()

    if results_df is None:
        st.warning("No comparison results found. Run notebooks/03_modeling.ipynb first.")
        return

    if metadata:
        st.success(f"Winner: **{metadata['model_name']}** ({metadata['imbalance_strategy']}) — test PR-AUC {metadata['test_pr_auc']:.3f}")

    metric_choice = st.selectbox("Metric to compare:", ["pr_auc", "recall", "roc_auc", "f1", "precision", "accuracy"], index=0)

    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model"] + " (" + plot_df["strategy"] + ")"
    plot_df = plot_df.sort_values(metric_choice)

    fig = px.bar(
        plot_df, x=metric_choice, y="label", orientation="h",
        color=metric_choice, color_continuous_scale="Greens",
        title=f"Cross-validated {metric_choice.upper()} by model and imbalance strategy",
    )
    st.plotly_chart(fig, width='stretch')

    st.dataframe(results_df.style.format({c: "{:.3f}" for c in results_df.select_dtypes(include=[np.number]).columns}))


def section_explainability():
    st.title("Explainability")
    explainer = get_explainer()
    pipeline = get_pipeline()

    if explainer is None or pipeline is None:
        st.warning("No saved model found. Run notebooks/03_modeling.ipynb first.")
        return

    _, X_test, _, y_test = get_test_split()

    st.subheader("Global importance")
    st.caption("Mean absolute SHAP value across a sample of the test set — recomputed once per session and cached.")

    @st.cache_data
    def compute_global_importance(_explainer, sample_index: tuple):
        rows = X_test.iloc[list(sample_index)]
        proba = pipeline.predict_proba(rows)[:, 1]
        contributions = []
        for i in range(len(rows)):
            for factor in _explainer.explain(rows.iloc[[i]], top_n=len(_explainer.feature_names)):
                contributions.append(factor)
        contrib_df = pd.DataFrame(contributions)
        return contrib_df.groupby("feature")["shap_value"].apply(lambda s: s.abs().mean()).sort_values(ascending=False)

    sample_size = min(100, len(X_test))
    sample_index = tuple(np.random.RandomState(get_settings_cached().random_seed).choice(len(X_test), sample_size, replace=False))
    global_importance = compute_global_importance(explainer, sample_index).head(15)

    fig = px.bar(
        x=global_importance.values[::-1], y=global_importance.index[::-1], orientation="h",
        labels={"x": "Mean |SHAP value| (log-odds)", "y": "Feature"},
        title="Global feature importance (SHAP)",
        color_discrete_sequence=["#1D9E75"],
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Explain an individual session")
    row_choice = st.slider("Test-set session index", 0, len(X_test) - 1, 0)
    row = X_test.iloc[[row_choice]]
    proba = float(pipeline.predict_proba(row)[:, 1][0])
    actual = "Purchase" if y_test.iloc[row_choice] == 1 else "No purchase"

    col1, col2 = st.columns(2)
    col1.metric("Predicted probability", f"{proba:.1%}")
    col2.metric("Actual outcome", actual)

    factors = explainer.explain(row, top_n=8)
    factors_df = pd.DataFrame(factors)
    factors_df["color"] = factors_df["direction"].map({"increases": "#1D9E75", "decreases": "#D85A30"})
    fig2 = go.Figure(go.Bar(
        x=factors_df["shap_value"], y=factors_df["feature"], orientation="h",
        marker_color=factors_df["color"],
    ))
    fig2.update_layout(title="Why this session got this prediction", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, width='stretch')


def section_metrics():
    st.title("Performance & system metrics")

    metadata = get_model_metadata()
    settings = get_settings_cached()

    if metadata is None:
        st.warning("No model metadata found. Run notebooks/03_modeling.ipynb first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Test PR-AUC", f"{metadata['test_pr_auc']:.3f}")
    col2.metric("Test ROC-AUC", f"{metadata['test_roc_auc']:.3f}")
    col3.metric("Decision threshold", f"{metadata['decision_threshold']:.4f}")

    st.warning(
        f"Without PageValues: PR-AUC drops to {metadata['test_pr_auc_no_pagevalues']:.3f} "
        f"— see docs/model_card.md for the deployment-timing assumption this model relies on."
    )

    pipeline = get_pipeline()
    if pipeline is not None:
        _, X_test, _, y_test = get_test_split()
        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= metadata["decision_threshold"]).astype(int)
        cm = pd.crosstab(y_test, preds, rownames=["Actual"], colnames=["Predicted"])

        st.subheader("Confusion matrix at chosen threshold")
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Greens",
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("System")
    st.write(f"Random seed: `{settings.random_seed}`")
    st.write(f"CV folds: `{settings.modeling.cv_folds}`")
    st.write(f"Model: `{metadata['model_name']}` ({metadata['imbalance_strategy']})")

    log_path = settings.paths.log_file
    if log_path.exists():
        st.subheader("Recent log activity")
        with log_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        st.code("".join(lines[-15:]) or "No log entries yet.", language="text")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("Purchase-Intent XAI")
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Live scoring", "Funnel & cohort analytics", "Model comparison", "Explainability", "Performance metrics"],
    )

    pages = {
        "Overview": section_overview,
        "Live scoring": section_live_scoring,
        "Funnel & cohort analytics": section_funnel_cohort,
        "Model comparison": section_model_comparison,
        "Explainability": section_explainability,
        "Performance metrics": section_metrics,
    }
    pages[page]()


if __name__ == "__main__":
    main()
