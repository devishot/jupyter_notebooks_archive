from datetime import datetime

import mlflow.sklearn
import requests

import pickle
import joblib
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow

from prefect import flow, task


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


@task
def get_year_month(run_date: str):
    date = datetime.strptime(run_date, "%Y-%m-%d")
    return {"year": date.year, "month": date.month}

@task
def get_next_date(year_month: dict) -> dict:
    year = year_month["year"]
    month = year_month["month"]

    if month == 12:
        return {"year": year + 1, "month": 1}
    else:
        return {"year": year, "month": month + 1}

@task
def download_file(year_month: dict, data_prefix: str) -> str:
    year = year_month["year"]
    month = year_month["month"]

    url = f'{NYC_TAXI_DATA_URI_PREFIX}/{data_prefix}_{year}-{month:02d}.parquet'
    output_path = f'{DATA_OUTPUT_PATH}/{data_prefix}_{year}-{month:02d}.parquet'

    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)

    return str(output_path)

@task
def read_dataframe(file_path: str, data_specific_fields: dict) -> pd.DataFrame:
    df = pd.read_parquet(file_path)

    print(file_path, data_specific_fields)
    print("number of rows loaded: ", len(df))

    df['duration'] = df[data_specific_fields['dropoff_time']] - df[data_specific_fields['pickup_time']]
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print("number of rows after data preparation: ", len(df))

    return df

@task
def prepare_features(df: pd.DataFrame, dv_path: str = None, output_x_path: str = "x_path"):
    # categorical = ['PU_DO']
    categorical = ['PULocationID', 'DOLocationID']
    # numerical = ['trip_distance']
    # dicts = df[categorical + numerical].to_dict(orient='records')
    dicts = df[categorical].to_dict(orient='records')

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

@task
def prepare_labels(df: pd.DataFrame, target: str = 'duration', output_y_path: str = "y_path"):
    y = df[target].values
    joblib.dump(y, output_y_path)

@task
def train_model(X_train_path, y_train_path, dv_path: str):
    X_train = joblib.load(X_train_path)
    y_train = joblib.load(y_train_path)

    with open(dv_path, "rb") as f:
        dv = pickle.load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)

        # Output intercept
        print(f"Intercept: {lr.intercept_}")
        mlflow.log_metric("intercept", lr.intercept_)

        rmse = root_mean_squared_error(y_train, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="model", registered_model_name="module-3-homework-nyc-yellow-taxi-duration-predictor")

        return run.info.run_id

@flow
def module_3_homework_train_pipeline(run_date: str = None):
    """
        The function’s return value is passed to the next task — no manual use of XComs required. 
        Under the hood, TaskFlow uses XComs to manage data passing automatically, 
        abstracting away the complexity of manual XCom management from the previous methods.
    """
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")

    year_month = get_year_month(run_date)

    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)

    curr_month_file_path = download_file(year_month, YELLOW_TAXI_DATA_PREFIX)

    df_train = read_dataframe(curr_month_file_path, YELLOW_TAXI_DATA_FIELDS)

    x_train_path = "models/x_train.bin"
    y_train_path = "models/y_train.bin"
    dv_path = "models/dv.pkl"

    prepare_features(df_train, output_x_path=x_train_path)

    prepare_labels(df_train, target='duration', output_y_path=y_train_path)

    train_model(x_train_path, y_train_path, dv_path=dv_path)


if __name__ == "__main__":
    module_3_homework_train_pipeline.serve(name="module-3-homework-deployment")