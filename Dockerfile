FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Download the model directly from GitHub LFS at build time
RUN curl -L \
  "https://media.githubusercontent.com/media/SwarnaRao24/credit-risk-model/main/credit_risk_model.joblib" \
  -o credit_risk_model.joblib

ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]