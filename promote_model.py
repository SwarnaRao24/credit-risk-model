"""
Promote the latest run to Staging, then to Production after validation.
Run manually or wire into your CI after a successful training run.
"""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "credit-risk-xgboost"

def register_and_promote():
    # 1. Get the latest run
    experiment = client.get_experiment_by_name("Credit_Risk_Model_Training")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    # 2. Register the model
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"Registered model version: {version}")

    # 3. Transition to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"Version {version} → Staging")

    return version


def promote_to_production(version: str):
    """Call this after validation passes."""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True  # archives previous production version
    )
    print(f"Version {version} → Production (previous versions archived)")


if __name__ == "__main__":
    v = register_and_promote()
    # After running your validation / A-B tests, call:
    # promote_to_production(v)