FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifact
COPY app.py .
COPY credit_risk_model.joblib .

# mlflow.db is NOT copied — it gets created at runtime
# or mounted via docker-compose volume

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]