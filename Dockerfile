FROM python:3.11-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces runs containers with restricted permissions.
# Create a non-root user (uid 1000 is what Spaces expects) and
# give it ownership of /app so the API can write drift logs.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy application code and model artifact
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser monitor.py .
COPY --chown=appuser:appuser credit_risk_model.joblib .

# Drift-monitoring inputs: the static training reference (committed),
# and a seed file so /predict has somewhere to append on first boot.
# Bulk training data (cs-training.csv) is intentionally excluded via
# .dockerignore — it's not needed at inference time.
COPY --chown=appuser:appuser data/reference.csv data/
COPY --chown=appuser:appuser data/recent_requests.csv data/

# mlflow.db is NOT copied — app falls back to the joblib artifact

USER appuser
ENV HOME=/home/appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]