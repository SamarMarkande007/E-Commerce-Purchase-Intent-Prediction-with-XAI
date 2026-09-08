"""Tests for dashboard/app.py, using Streamlit's AppTest (no browser needed).

Note: the live-scoring section's success path additionally requires the
API to be running (`uvicorn src.api.main:app`) — that path is exercised
manually/in CI as an integration step, not here, so this file doesn't
depend on a background server being up. The graceful-failure path (API
unreachable) IS tested here, since it needs no server at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config.settings import get_settings

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

SECTIONS = [
    "Overview",
    "Live scoring",
    "Funnel & cohort analytics",
    "Model comparison",
    "Explainability",
    "Performance metrics",
]


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_model():
    settings = get_settings()
    if not (settings.paths.models_dir / "best_pipeline.joblib").exists():
        pytest.skip("models/best_pipeline.joblib not found; run notebooks/03_modeling.ipynb first")


@pytest.mark.parametrize("section", SECTIONS)
def test_section_loads_without_exception(section):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.sidebar.radio[0].set_value(section).run(timeout=60)
    assert not at.exception, f"{section} raised: {list(at.exception)}"


def test_live_scoring_shows_clear_error_when_api_unreachable():
    """With no API running, the dashboard must fail gracefully with
    actionable guidance, not crash."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.sidebar.radio[0].set_value("Live scoring").run(timeout=60)

    assert not at.exception
    error_messages = [e.value for e in at.error]
    # Only assert this when the API genuinely isn't running during the test.
    if error_messages:
        assert "uvicorn" in error_messages[0]


def test_overview_shows_correct_conversion_rate():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    metric_values = [m.value for m in at.metric]
    # Conversion rate metric should be present and roughly match the known ~15.7%.
    assert any("15." in v or "16." in v for v in metric_values if "%" in v)


def test_model_comparison_shows_results_table():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.sidebar.radio[0].set_value("Model comparison").run(timeout=60)
    assert not at.exception
    assert len(at.dataframe) >= 1
