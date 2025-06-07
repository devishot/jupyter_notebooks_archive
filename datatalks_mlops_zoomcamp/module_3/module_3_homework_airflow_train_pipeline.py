import textwrap
from datetime import datetime, timedelta

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

# Operators; we need this to operate!
from airflow.providers.standard.operators.bash import PythonOperator

# The DAG object; we'll need this to instantiate a DAG
from airflow.sdk import dag, task


def get_next_date(year: int, month: int):
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1

DATA_YEAR = 2025
DATA_MONTH = 1

@dag(
    schedule=None,
    start_date=datetime(2025, 6, 6),
    catchup=False,
    tags=["mlops", "zoomcamp", "homework"],
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
) 
def module_3_homework_airflow_train_pipeline():
    @task()
    def setup_mlflow():
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("mlops-zoomcamp-module-3-homework-airflow")

        models_folder = Path("models")
        models_folder.mkdir(exist_ok=True)

    """
        The function’s return value is passed to the next task — no manual use of XComs required. 
        Under the hood, TaskFlow uses XComs to manage data passing automatically, 
        abstracting away the complexity of manual XCom management from the previous methods.
    """
    @task()
    def read_dataframe(year: int, month: int) -> pd.DataFrame:
        file_path = f'../data/green_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(file_path)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

        return df

    """
        The use of `@task(multiple_outputs=True)` above — this tells Airflow that the function returns a dictionary of values that should be split into individual XComs. 
        Each key in the returned dictionary becomes its own XCom entry, which makes it easy to reference specific values in downstream tasks. 
        If you omit `multiple_outputs=True`, the entire dictionary is stored as a single XCom instead, and must be accessed as a whole.
    """
    @task(multiple_outputs=True)
    def create_X(df: pd.DataFrame, dv: DictVectorizer=None) -> dict:
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')

        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)

        return {"x": X, "dv": dv}

    @task()
    def train_model(X_train, y_train, X_val, y_val, dv: DictVectorizer):
        with mlflow.start_run() as run:
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            best_params = {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:linear',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }

            mlflow.log_params(best_params)

            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

            return run.info.run_id
    
    year, month = DATA_YEAR, DATA_MONTH
    next_year, next_month = get_next_date(year, month)

    df_train = read_dataframe(year=year, month=month)
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    return run_id
module_3_homework_airflow_train_pipeline()