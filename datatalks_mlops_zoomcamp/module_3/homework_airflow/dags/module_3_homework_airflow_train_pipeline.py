from datetime import datetime, timedelta

import requests

import pickle
import joblib
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

from airflow.decorators import dag, task

MLFLOW_EXPERIMENT_NAME = "mlops-zoomcamp-module-3-homework-airflow"
MLFLOW_TRACKING_URI = "http://localhost:5000"

NYC_TAXI_DATA_URI_PREFIX = "https://d37ci6vzurychx.cloudfront.net/trip-data"

DATA_OUTPUT_PATH = "data"

GREEN_TAXI_DATA_PREFIX = "green_tripdata"
YELLOW_TAXI_DATA_PREFIX = "yellow_tripdata"

GREEN_TAXI_DATA_FIELDS = {
    'pickup_time': "lpep_pickup_datetime",
    'dropoff_time': "lpep_dropoff_datetime"
}

YELLOW_TAXI_DATA_FIELDS = {
    'pickup_time': "tpep_pickup_datetime",
    'dropoff_time': "tpep_dropoff_datetime"
}

@dag(
    schedule="@monthly",
    start_date=datetime(2025, 6, 6),
    catchup=False,
    tags=["mlops", "zoomcamp", "homework"],
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
) 
def module_3_homework_airflow_train_pipeline():
    """
        The function’s return value is passed to the next task — no manual use of XComs required. 
        Under the hood, TaskFlow uses XComs to manage data passing automatically, 
        abstracting away the complexity of manual XCom management from the previous methods.
    """
    @task()
    def get_year_month(ds=None):
        date = datetime.strptime(ds, "%Y-%m-%d")  # convert ds string to datetime
        return {"year": date.year, "month": date.month}

    @task()
    def get_next_date(year_month: dict) -> dict:
        year = year_month["year"]
        month = year_month["month"]

        if month == 12:
            return {"year": year + 1, "month": 1}
        else:
            return {"year": year, "month": month + 1}

    @task()
    def download_file(year_month: dict, data_prefix):
        year = year_month["year"]
        month = year_month["month"]

        url = f'{NYC_TAXI_DATA_URI_PREFIX}/{data_prefix}_{year}-{month:02d}.parquet'
        output_path = f'{DATA_OUTPUT_PATH}/{data_prefix}_{year}-{month:02d}.parquet'

        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)

        return str(output_path)

    @task()
    def read_dataframe(file_path: str, data_specific_fields: dict) -> pd.DataFrame:
        df = pd.read_parquet(file_path)

        print(data_specific_fields)

        df['duration'] = df[data_specific_fields['dropoff_time']] - df[data_specific_fields['pickup_time']] #df.lpep_dropoff_datetime - df.lpep_pickup_datetime
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
    @task()
    def prepare_features(df: pd.DataFrame, dv_path: str = None, output_x_path: str = "x_path"):
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')

        if dv_path is None:
            dv = DictVectorizer()
            X = dv.fit_transform(dicts)
            with open("models/dv.pkl", "wb") as f_out:
                pickle.dump(dv, f_out)
        else:
            with open(dv_path, "rb") as f_in:
                dv = pickle.load(f_in)
                X = dv.transform(dicts)
        
        joblib.dump(X, output_x_path)

    @task()
    def prepare_labels(df: pd.DataFrame, target: str = 'duration', output_y_path: str = "y_path"):
        y = df[target].values
        joblib.dump(y, output_y_path)

    @task()
    def train_model(X_train_path, y_train_path, X_val_path, y_val_path, dv_path):
        X_train = joblib.load(X_train_path)
        y_train = joblib.load(y_train_path)
        X_val = joblib.load(X_val_path)
        y_val = joblib.load(y_val_path)

        with open(dv_path, "rb") as f:
            dv = pickle.load(f)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

    year_month = get_year_month()
    next_year_month = get_next_date(year_month)

    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    curr_month_file_path = download_file(year_month, YELLOW_TAXI_DATA_PREFIX)
    next_month_file_path = download_file(next_year_month, YELLOW_TAXI_DATA_PREFIX)

    df_train = read_dataframe(curr_month_file_path, YELLOW_TAXI_DATA_FIELDS)
    df_val = read_dataframe(next_month_file_path, YELLOW_TAXI_DATA_FIELDS)

    x_train_path = "models/x_train.bin"
    y_train_path = "models/y_train.bin"
    x_val_path = "models/x_val.bin"
    y_val_path = "models/y_val.bin"
    dv_path = "models/dv.pkl"

    prepare_features(df_train, output_x_path=x_train_path)
    prepare_features(df_val, dv_path=dv_path, output_x_path=x_val_path)

    prepare_labels(df_train, target='duration', output_y_path=y_train_path)
    prepare_labels(df_val, target='duration', output_y_path=y_val_path)

    train_model(x_train_path, y_train_path, x_val_path, y_val_path, dv_path=dv_path)

module_3_homework_airflow_train_pipeline()