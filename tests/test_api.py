"""Tests for src/api/main.py, using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config.settings import get_settings

VALID_SESSION = {
    "Administrative": 2, "Administrative_Duration": 40.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 25, "ProductRelated_Duration": 620.5,
    "BounceRates": 0.01, "ExitRates": 0.02, "PageValues": 15.0,
    "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
    "Browser": 2, "Region": 1, "TrafficType": 2,
    "VisitorType": "Returning_Visitor", "Weekend": False,
}


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    if not (settings.paths.models_dir / "best_pipeline.joblib").exists():
        pytest.skip("models/best_pipeline.joblib not found; run notebooks/03_modeling.ipynb first")

    from src.api.main import app
    with TestClient(app) as test_client:  # triggers the lifespan startup/shutdown
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "catboost"
    assert 0 <= body["decision_threshold"] <= 1


def test_predict_valid_session_returns_full_response(client):
    response = client.post("/predict", json=VALID_SESSION)
    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["purchase_probability"] <= 1
    assert isinstance(body["will_purchase"], bool)
    assert body["will_purchase"] == (body["purchase_probability"] >= body["decision_threshold"])
    assert body["confidence"] in {"low", "medium", "high"}
    assert len(body["top_factors"]) == 5
    for factor in body["top_factors"]:
        assert factor["direction"] in {"increases", "decreases"}


def test_predict_rejects_invalid_month(client):
    bad_session = {**VALID_SESSION, "Month": "NotAMonth"}
    response = client.post("/predict", json=bad_session)
    assert response.status_code == 422


def test_predict_rejects_negative_count(client):
    bad_session = {**VALID_SESSION, "ProductRelated": -5}
    response = client.post("/predict", json=bad_session)
    assert response.status_code == 422


def test_predict_rejects_missing_fields(client):
    response = client.post("/predict", json={"Administrative": 2})
    assert response.status_code == 422


def test_predict_high_pagevalues_session_favors_purchase(client):
    """Sanity check tying the API back to the model card's core finding:
    a session with strong PageValues should score meaningfully higher
    than one with none, all else equal."""
    high_pv_session = {**VALID_SESSION, "PageValues": 50.0}
    low_pv_session = {**VALID_SESSION, "PageValues": 0.0}

    high_response = client.post("/predict", json=high_pv_session).json()
    low_response = client.post("/predict", json=low_pv_session).json()

    assert high_response["purchase_probability"] > low_response["purchase_probability"]
