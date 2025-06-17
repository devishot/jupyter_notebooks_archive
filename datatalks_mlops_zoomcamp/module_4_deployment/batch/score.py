#!/usr/bin/env python
# coding: utf-8

import os
import sys
import uuid
import pandas as pd
import mlflow

# # temporary solution to see environment variables when running Jupyter Notebook from IDE
# os.environ["MLFLOW_EXPERIMENT_NAME"] = "module_4_nyc_taxi_duration_prediction_with_sklearn_pipeline_experiment"
# os.environ["RUN_ID"] = "8d6b2c8289b94d1cb40a313e0bf92aca"

# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
# os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_password"


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
print("Environment variables for MLFlow: ", MLFLOW_TRACKING_URI)

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
print("Environment variables for S3: ", MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
RUN_ID = os.environ.get("RUN_ID")
print("Environment variables for model: ", MLFLOW_EXPERIMENT_NAME, RUN_ID)


def generate_ride_ids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['ride_id'] = generate_ride_ids(len(df))

    return df

def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):
    model_uri = f's3://bucket/1/{run_id}/artifacts/model'
    print(f'Loading model using uri: {model_uri}')
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def prepare_results(df, y_pred):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['lpep_dropoff_datetime'] = df['lpep_dropoff_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['trip_distance'] = df['trip_distance']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    return df

def apply_model(input_file, run_id, output_file):
    print(f'Reading data from file: {input_file}')

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'Loading model with run_id: {run_id}')
    model = load_model(run_id)

    print('Applying model...')
    y_pred = model.predict(dicts)

    print(f'Saving results to file: {output_file}')
    df_result = prepare_results(df, y_pred)
    df_result.to_parquet(output_file, engine='pyarrow', index=False)


def run():
    taxi_type = sys.argv[1] # green or yellow
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 1

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/prediction_{year:04d}-{month:02d}.parquet'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    apply_model(
        input_file=input_file, 
        run_id=RUN_ID, 
        output_file=output_file
    )

if __name__ == '__main__':
    run()
