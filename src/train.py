from __future__ import annotations

import json, os
import pickle
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from helper_fun import AlsModel
from dotenv import load_dotenv

load_dotenv()



def train_and_log_to_mlflow(
    interactions: pd.DataFrame,
    mlflow_tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
) -> dict:
    """
    interactions columns:
        user_id, item_id, weight
    """
    from sklearn.preprocessing import LabelEncoder

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_registry_uri(mlflow_tracking_uri)
    # mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # 0. split
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="ms")


    # split_last_month_for_test = pd.Timestamp.today() - pd.DateOffset(months=1)
    split_point = pd.to_datetime("2015-09-01")
    split_idx = interactions["timestamp"] > split_point

    events_train = interactions[~split_idx]
    events_test = interactions[split_idx]

    # 1. encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    events_train["user_idx"] = user_encoder.fit_transform(events_train["visitorid"])
    events_train["item_idx"] = item_encoder.fit_transform(events_train["itemid"])

    weights = {
    "view": 1,
    "addtocart": 5,
    "transaction": 3
    }

    events_train["weight"] = events_train["event"].map(weights)

    # 2. sparse matrix
    user_items = sp.csr_matrix(
        (
            events_train["weight"].astype(float),
            (events_train["user_idx"], events_train["item_idx"]),
        )
    )

    # 3. popular fallback
    interactions["weight"] = interactions["event"].map(weights)
    popular_items = (
        interactions.groupby("itemid", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .rename(columns={"weight": "popularity"})
    )

    # 4. train ALS
    factors = 64
    regularization = 0.09
    iterations = 10
    alpha = 40

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )

    model.fit((user_items * alpha))

    events_test = events_test[events_test["visitorid"].isin(user_encoder.classes_) & events_test["itemid"].isin(item_encoder.classes_)].copy()
    sample_user = events_test.sample(1)

    als_recommendations = get_als_recommendations(user_items, model, sample_user["visitorid"], user_encoder, item_encoder, False, 15)

    # 5. metrics
    relevant_items = events_test[events_test["visitorid"] == sample_user["visitorid"].iloc[0]]["itemid"].tolist()

    als_recall_score = len(set(als_recommendations["item_id"].tolist()) & set(relevant_items)) / len(relevant_items)
    als_precision_score = len(set(als_recommendations["item_id"].tolist()) & set(relevant_items)) / len(als_recommendations["item_id"].tolist())

    metrics = {
        "precision_at_k": als_precision_score,
        "recall_at_k": als_recall_score,
    }

    params = {
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha": alpha,
        "num_users": int(user_items.shape[0]),
        "num_items": int(user_items.shape[1]),
    }

    custom_model = AlsModel(model)

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY")

    with mlflow.start_run(run_name="als_retrain", experiment_id=experiment_id) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            with open(tmp / "als_model.pkl", "wb") as f:
                pickle.dump(model, f)

            with open(tmp / "user_encoder.pkl", "wb") as f:
                pickle.dump(user_encoder, f)

            with open(tmp / "item_encoder.pkl", "wb") as f:
                pickle.dump(item_encoder, f)

            sp.save_npz(tmp / "user_items_matrix.npz", user_items)
            popular_items.to_parquet(tmp / "popular_items.parquet", index=False)

            with open(tmp / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            mv = mlflow.pyfunc.log_model(python_model=custom_model, 
                                    artifact_path="models", 
                                    registered_model_name=registered_model_name,
                                    artifacts={
                                        "als_model": str(tmp / "als_model.pkl"),
                                        "user_encoder": str(tmp / "user_encoder.pkl"),
                                        "item_encoder": str(tmp / "item_encoder.pkl"),
                                        "user_items_matrix": str(tmp / "user_items_matrix.npz"),
                                        "popular_items": str(tmp / "popular_items.parquet"),
                                    },
                                    pip_requirements=[
                                        "mlflow",
                                        "pandas",
                                        "scipy",
                                        "implicit",
                                        "pyarrow",
                                        "scikit-learn"
                                    ])

    return {
        "run_id": run.info.run_id,
        "metrics": metrics,
    }


def get_als_recommendations(user_item_matrix, model, user_id, user_encoder, item_encoder, include_seen=True, n=5):
    user_id_enc = user_encoder.transform(user_id)[0]
    recommendations = model.recommend(user_id_enc,
                                      user_item_matrix[user_id_enc],
                                      filter_already_liked_items=include_seen,
                                      N=n)
    recommendations = pd.DataFrame({"item_id_enc": recommendations[0], "score": recommendations[1]})
    recommendations['item_id'] = item_encoder.inverse_transform(recommendations["item_id_enc"])

    return recommendations