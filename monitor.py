"""
Run periodically to check for data drift between training reference
data and recent API requests.

Usage:
    python monitor.py

Expects:
    data/reference.csv         — created once by create_reference.py
    data/recent_requests.csv   — auto-appended by /predict endpoint
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.pipeline.column_mapping import ColumnMapping
import json
import os
from datetime import datetime

FEATURE_COLS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
    'TotalTimesLate', 'IncomePerPerson'
]


def run_drift_report(
    reference_path: str = "data/reference.csv",
    current_path: str = "data/recent_requests.csv",
    output_path: str = "drift_report.html"
):
    # ── Guard: check files exist before doing anything ────
    if not os.path.exists(reference_path):
        print(f"ERROR: Reference file not found at '{reference_path}'.")
        print("Run create_reference.py first to generate it.")
        return None

    if not os.path.exists(current_path):
        print(f"ERROR: No recent requests file found at '{current_path}'.")
        print("Make a few POST /predict calls first — each one logs a row.")
        return None

    # ── Load reference (has headers from create_reference.py) ────
    reference = pd.read_csv(reference_path)[FEATURE_COLS].dropna()

    # ── Load current (no header — app.py appends without writing column names) ──
    current = pd.read_csv(current_path, header=None, names=FEATURE_COLS).dropna()

    # ── Guard: need at least 30 rows to get meaningful drift stats ────
    if len(current) < 30:
        print(f"WARNING: Only {len(current)} request(s) logged so far.")
        print("Make at least 30 POST /predict calls before running drift detection.")
        return None

    print(f"Reference rows : {len(reference)}")
    print(f"Current rows   : {len(current)}")

    # ── Build report ──────────────────────────────────────
    column_mapping = ColumnMapping(numerical_features=FEATURE_COLS)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    report.save_html(output_path)
    print(f"Report saved → {output_path}")

    # ── Extract drift result safely ───────────────────────
    result = report.as_dict()

    try:
        drift_detected = result["metrics"][0]["result"]["dataset_drift"]
        drifted_features = result["metrics"][0]["result"]["number_of_drifted_columns"]
        total_features = result["metrics"][0]["result"]["number_of_columns"]
    except (KeyError, IndexError):
        drift_detected = False
        drifted_features = 0
        total_features = len(FEATURE_COLS)

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "current_rows_checked": len(current),
        "report_path": output_path,
    }

    print("\n" + json.dumps(summary, indent=2))

    if drift_detected:
        print(f"\nDRIFT DETECTED — {drifted_features}/{total_features} features drifted.")
        print("Consider retraining your model or investigating the incoming data.")
    else:
        print(f"\nNo drift detected — model inputs look stable.")

    return summary


if __name__ == "__main__":
    run_drift_report(
        reference_path="data/reference.csv",
        current_path="data/recent_requests.csv",
        output_path="drift_report.html"
    )