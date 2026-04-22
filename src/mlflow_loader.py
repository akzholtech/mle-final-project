from pathlib import Path

import mlflow, os
from mlflow.artifacts import download_artifacts

from src.recommender import ALSRecommender


def load_recommender_from_mlflow() -> ALSRecommender:
    TRACKING_URL = "http://127.0.0.1:5000"
    MODEL_URI = "models:/Recsys_model/latest"
    mlflow.set_tracking_uri(TRACKING_URL)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY")

    local_model_dir = download_artifacts(MODEL_URI)
    return ALSRecommender.from_local_dir(Path(local_model_dir))