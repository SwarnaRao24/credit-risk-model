from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os
import time
import platform
from mlflow import MlflowClient

# ── Initialize FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Inference API",
    description="XGBoost credit default predictor with MLflow tracking",
    version="1.0.0",
)

START_TIME = time.time()

# ── Load Model: MLflow DB → joblib fallback ─────────────────────────────────
print("Connecting to MLflow database...")
MODEL_SOURCE = "unknown"
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

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Loading model from Run ID: {run_id}...")
    model = mlflow.pyfunc.load_model(model_uri)
    MODEL_SOURCE = "mlflow_db"
    print("✅ MLflow model loaded successfully.")

except Exception as e:
    print(f"⚠️  MLflow route failed: {e}")
    print("Falling back to joblib model...")
    import joblib
    model = joblib.load("credit_risk_model.joblib")
    MODEL_SOURCE = "joblib_fallback"
    print("✅ Fallback model loaded.")

# ── Feature Schema ───────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalTimesLate",
    "IncomePerPerson",
]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Production Credit Risk API is live and running!",
        "interactive_docs": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health_check():
    """
    Ping this endpoint to verify the API and model are fully operational.
    Returns uptime, model source, and status — ideal for CI smoke tests
    and Hugging Face Spaces liveness probes.
    """
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_status": "loaded" if model is not None else "FAILED",
        "model_source": MODEL_SOURCE,
        "python_version": platform.python_version(),
        "feature_count": len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict_credit_risk(applicant_data: dict):
    try:
        # On-the-fly feature engineering
        applicant_data["TotalTimesLate"] = (
            int(applicant_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0))
            + int(applicant_data.get("NumberOfTimes90DaysLate", 0))
            + int(applicant_data.get("NumberOfTime60-89DaysPastDueNotWorse", 0))
        )
        monthly_income = float(applicant_data.get("MonthlyIncome", 0))
        dependents = int(applicant_data.get("NumberOfDependents", 0))
        applicant_data["IncomePerPerson"] = monthly_income / (dependents + 1)

        df = pd.DataFrame([applicant_data])[FEATURE_NAMES]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        prediction = int(model.predict(df)[0])

        return {
            "default_prediction": prediction,
            "status": "High Risk of Default" if prediction == 1 else "Approved / Low Risk",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))