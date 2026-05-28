# Real-Time Credit Risk Inference Engine & MLOps Pipeline

An enterprise-grade, production-ready machine learning pipeline that trains an XGBoost classifier to assess credit default risk and serves predictions via a high-performance FastAPI backend. The architecture shifts away from traditional static file dependencies by dynamically querying an MLflow tracking database to serve the latest model version at runtime.

##️ Tech Stack & Trending Tools Used
* **Backend Framework:** FastAPI (Asynchronous Python Web Framework)
* **Machine Learning Framework:** XGBoost (Extreme Gradient Boosting)
* **MLOps & Experiment Tracking:** MLflow (Tracking Server with SQLite storage backend)
* **Data Engineering & Analysis:** Pandas, Jupyter Notebooks
* **Server Gateway:** Uvicorn (ASGI Server implementation)
* **Version Control:** Git & GitHub (Optimized for clean artifact management)

---

##  System Architecture


```mermaid
graph TD
    classDef client fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px,color:#000;
    classDef server fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef database fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#000;
    classDef model fill:#ffebee,stroke:#f44336,stroke-width:2px,color:#000;

    A[Client Web Request<br><i>Raw JSON Payload</i>]:::client -->|HTTP POST /predict| B(FastAPI Server Framework):::server
    
    subgraph Data Pipeline Engine
        B --> C{Feature Engineering<br>Layer}:::server
        C -->|Compute| D[TotalTimesLate]:::server
        C -->|Compute| E[IncomePerPerson]:::server
    end

    B -->|Query Active Run Hash| F[(SQLite MLflow Registry<br><i>mlflow.db</i>)]:::database
    F -->|Return Model Artifact Blueprint| B

    D & E & B -->|Pass Schema-Aligned Matrix| G[XGBoost Inference Engine]:::model
    G -->|Return Real-Time Assessment| H[JSON Response<br><i>High/Low Default Risk</i>]:::client
```

---

## Local Deployment Guide

### 1. Environment Setup
Clone the repository and install the production dependencies:
```bash
pip install -r requirements.txt #Install Dependencies

mlflow ui --backend-store-uri sqlite:///mlflow.db  #To launch MLFlow experiments UI

uvicorn app:app --reload   #Run server

#To send a mock credit application payload
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d '{
  "RevolvingUtilizationOfUnsecuredLines": 0.85,
  "age": 42,
  "NumberOfTime30-59DaysPastDueNotWorse": 2,
  "DebtRatio": 0.55,
  "MonthlyIncome": 4500,
  "NumberOfOpenCreditLinesAndLoans": 6,
  "NumberOfTimes90DaysLate": 1,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}'
```
### 2.API Expected Output
```bash
#JSON
{
  "default_prediction": 1,
  "status": "High Risk of Default"
}
```

---
**Developer:** Swarna Rao  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swarnamukhirchintalapudi)

**Focus:** Data Science | AI Strategy | Finance
