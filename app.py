from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import time
import logging
import shap

from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Inference API",
    description="Production XGBoost credit default risk scorer with MLflow + SHAP",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────
# We load via mlflow.xgboost.load_model() (not pyfunc) so we get
# the native XGBClassifier object — which gives us predict_proba + SHAP
xgb_model = None
shap_explainer = None

logger.info("Connecting to MLflow database...")
try:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment = client.get_experiment_by_name("Credit_Risk_Model_Training")
    if experiment is None:
        raise ValueError("Experiment 'Credit_Risk_Model_Training' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise ValueError("No training runs found.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading XGBClassifier from Run ID: {run_id}")

    # This gives us the native XGBClassifier — has predict_proba + works with SHAP
    xgb_model = mlflow.xgboost.load_model(model_uri)
    logger.info("XGBClassifier loaded from MLflow.")

except Exception as e:
    logger.warning(f"MLflow route failed: {e}")
    logger.info("No mlflow.db found in container — loading from joblib fallback.")
    import joblib
    xgb_model = joblib.load("credit_risk_model.joblib")
    logger.info("Fallback model loaded successfully from credit_risk_model.joblib.")

# ── SHAP explainer (runs once at startup) ─────────────────
try:
    shap_explainer = shap.TreeExplainer(xgb_model)
    logger.info("SHAP TreeExplainer initialized.")
except Exception as e:
    shap_explainer = None
    logger.warning(f"SHAP init failed: {e}")

# ── Feature config ────────────────────────────────────────
FEATURE_NAMES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'TotalTimesLate',
    'IncomePerPerson'
]

# ── Request schema ─────────────────────────────────────────
class CreditApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=110)
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfDependents: int = Field(..., ge=0)

    # Aliased fields — hyphens are not valid Python attr names
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")

    model_config = {"populate_by_name": True}


# ── Helpers ───────────────────────────────────────────────
def classify_risk(proba: float) -> dict:
    if proba >= 0.60:
        return {"tier": "High",   "status": "High Risk of Default",          "action": "Declined"}
    elif proba >= 0.35:
        return {"tier": "Review", "status": "Moderate Risk — Manual Review", "action": "Under Review"}
    else:
        return {"tier": "Low",    "status": "Approved — Low Risk",           "action": "Approved"}


def build_features(data: dict) -> pd.DataFrame:
    """Engineer TotalTimesLate + IncomePerPerson, return aligned DataFrame."""
    data["TotalTimesLate"] = (
        int(data.get("NumberOfTime30-59DaysPastDueNotWorse", 0))
        + int(data.get("NumberOfTimes90DaysLate", 0))
        + int(data.get("NumberOfTime60-89DaysPastDueNotWorse", 0))
    )
    monthly_income = float(data.get("MonthlyIncome", 0))
    dependents = int(data.get("NumberOfDependents", 0))
    data["IncomePerPerson"] = monthly_income / (dependents + 1)

    df = pd.DataFrame([data])[FEATURE_NAMES]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def log_request(df: pd.DataFrame):
    """Append this prediction's features to CSV for drift monitoring."""
    df.to_csv("data/recent_requests.csv", mode="a", header=False, index=False)


# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Credit Risk API v2.0 is live.", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": xgb_model is not None,
        "shap_available": shap_explainer is not None,
        "timestamp": time.time(),
    }


@app.post("/predict")
def predict(application: CreditApplication):
    try:
        start = time.time()
        raw = application.model_dump(by_alias=True)
        df = build_features(raw)

        prediction = int(xgb_model.predict(df)[0])
        proba = float(xgb_model.predict_proba(df)[0][1])
        risk = classify_risk(proba)

        # Log for drift monitoring (won't crash if data/ doesn't exist yet)
        try:
            log_request(df)
        except Exception:
            pass

        return {
            "default_prediction": prediction,
            "default_probability": round(proba, 4),
            "risk_tier": risk["tier"],
            "status": risk["status"],
            "action": risk["action"],
            "engineered_features": {
                "TotalTimesLate": int(df["TotalTimesLate"].iloc[0]),
                "IncomePerPerson": round(float(df["IncomePerPerson"].iloc[0]), 2),
            },
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain")
def explain(application: CreditApplication):
    """Returns the top 5 features driving this specific prediction."""
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available.")
    try:
        raw = application.model_dump(by_alias=True)
        df = build_features(raw)

        shap_values = shap_explainer.shap_values(df)
        # XGBClassifier TreeExplainer returns shape (n_samples, n_features)
        values = shap_values[0]

        factors = sorted(
            zip(FEATURE_NAMES, values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        return {
            "top_risk_factors": [
                {
                    "feature": feat,
                    "shap_value": round(float(val), 4),
                    "direction": "increases_risk" if val > 0 else "reduces_risk",
                }
                for feat, val in factors
            ],
            "note": "Positive SHAP = pushes toward default. Negative = pushes toward approval."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))