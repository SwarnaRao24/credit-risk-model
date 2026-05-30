# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY credit_risk_model.joblib .

# Hugging Face Spaces runs containers as non-root user on port 7860
# FastAPI will fall back to the joblib model (no mlflow.db in container)
ENV PORT=7860

EXPOSE 7860

# Start the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]