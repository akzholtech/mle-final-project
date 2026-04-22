from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
from airflow.decorators import dag, task

from src.data_loader import load_interactions
from src.train import train_and_log_to_mlflows



@dag(
    dag_id="als_retrain_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule="0 3 * * 1",
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 2,
        "retry_delay": timedelta(minutes=10)
    },
    tags=["recsys", "als", "mlflow"]
)
def als_retrain_pipeline():
    @task
    def extract():
        df = load_interactions()
        return df
    
    @task
    def train(data: pd.DataFrame):
        
        interactions = data

        result = train_and_log_to_mlflows(
            data,
            "http://127.0.0.1:5000",
            "Training_RECSYS_model",
            "Recsys_model"
        )
        return result
    
    @task
    def report(result: dict):
        print("Train completed")
        print(result)
    
    records = extract()
    result = train(records)
    report(result)

als_retrain_pipeline()