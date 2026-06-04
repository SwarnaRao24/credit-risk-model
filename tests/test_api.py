from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# ── Mock the model BEFORE app.py is imported ──────────────
# This prevents app.py from trying to load mlflow.db or joblib
# during the CI environment where those files don't exist.

mock_xgb = MagicMock()
mock_xgb.predict.return_value = np.array([1])
mock_xgb.predict_proba.return_value = np.array([[0.2, 0.8]])

mock_shap_explainer = MagicMock()
mock_shap_explainer.shap_values.return_value = np.array([
    [0.42, -0.1, 0.31, 0.08, -0.05, 0.02, 0.19, 0.01, 0.07, -0.03, 0.22, -0.08]
])

with patch("mlflow.set_tracking_uri"), \
     patch("mlflow.MlflowClient"), \
     patch("mlflow.xgboost.load_model", return_value=mock_xgb), \
     patch("shap.TreeExplainer", return_value=mock_shap_explainer):
    from app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 40,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.3,
    "MonthlyIncome": 6000,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_expected_keys():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "default_prediction" in data
    assert "default_probability" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["Low", "Review", "High"]


def test_predict_probability_range():
    r = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    prob = r.json()["default_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_invalid_age():
    bad = {**SAMPLE_PAYLOAD, "age": 10}  # age < 18 should fail validation
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_explain_returns_shap_factors():
    r = client.post("/explain", json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "top_risk_factors" in data
    assert len(data["top_risk_factors"]) == 5
    assert "feature" in data["top_risk_factors"][0]
    assert "shap_value" in data["top_risk_factors"][0]
    assert "direction" in data["top_risk_factors"][0]