from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import os

from mlflow import MlflowClient

# 1. Initialize FastAPI
app = FastAPI(title="Production Credit Risk Inference API")

# 2. DYNAMICALLY QUERY THE SQLITE DATABASE FOR THE LATEST RUN
print("Connecting to local MLflow database...")
try:
    # Tell MLflow to look at your local sqlite database file
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    # Fetch your experiment details by name
    experiment = client.get_experiment_by_name("Credit_Risk_Model_Training")
    if experiment is None:
        raise ValueError(
            "Experiment 'Credit_Risk_Model_Training' not found in database."
        )

    # Query the database for the most recent successful run in this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise ValueError("No training runs found inside this experiment.")

    # Extract the exact model path location dynamically from the database record
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Database match found! Loading model artifacts from Run ID: {run_id}...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("🚀 Success! Best model dynamically loaded from MLflow database into memory.")

except Exception as e:
    print(f"❌ MLflow database route failed: {str(e)}")
    print("Engaging backup safe-mode: Loading static joblib fallback file...")
    import joblib

    model = joblib.load("credit_risk_model.joblib")
    print("Backup model loaded successfully.")

# 3. Your 12 notebook features
# Temporary alignment list to match your logged MLflow run schema
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


@app.get("/")
def home():
    return {"message": "Production Credit Risk API is live and running!"}


@app.post("/predict")
def predict_credit_risk(applicant_data: dict):
    try:
        # 2. DYNAMICALLY ENGINEER THE FEATURES ON THE FLY FOR INCOMING DATA
        # Calculate TotalTimesLate
        applicant_data["TotalTimesLate"] = (
            int(applicant_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0))
            + int(applicant_data.get("NumberOfTimes90DaysLate", 0))
            + int(applicant_data.get("NumberOfTime60-89DaysPastDueNotWorse", 0))
        )

        # Calculate IncomePerPerson safely to avoid division by zero
        monthly_income = float(applicant_data.get("MonthlyIncome", 0))
        dependents = int(applicant_data.get("NumberOfDependents", 0))
        applicant_data["IncomePerPerson"] = monthly_income / (dependents + 1)

        # Convert to DataFrame and sort columns to match model blueprint
        df = pd.DataFrame([applicant_data])
        df = df[FEATURE_NAMES]

        # Ensure data types match expected numeric inputs for XGBoost
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        prediction_raw = model.predict(df)[0]
        prediction = int(prediction_raw)

        return {
            "default_prediction": prediction,
            "status": "High Risk of Default"
            if prediction == 1
            else "Approved / Low Risk",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
