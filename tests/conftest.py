import os
import mlflow


def pytest_configure(config):
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
