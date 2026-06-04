"""
Run it once: python create_reference.py
Creates data/reference.csv — a 10,000 row sample of training data
used as the baseline for Evidently drift detection.
"""
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Load your existing training data
df = pd.read_csv("data/cs-training.csv")

# Engineer the same features your model uses
df["TotalTimesLate"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
    + df["NumberOfTimes90DaysLate"].fillna(0)
    + df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
)
df["IncomePerPerson"] = df["MonthlyIncome"].fillna(0) / (df["NumberOfDependents"].fillna(0) + 1)

FEATURE_NAMES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
    'TotalTimesLate', 'IncomePerPerson'
]

reference = df[FEATURE_NAMES].dropna().sample(n=10000, random_state=42)
reference.to_csv("data/reference.csv", index=False)
print(f"reference.csv created — {len(reference)} rows, {len(reference.columns)} columns.")