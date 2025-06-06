#!/usr/bin/env python
# coding: utf-8

"""
file being converted to a python file from a jupyter notebook using command:
> jupyter nbconvert --to=script module_3_duration_train_with_best_features.ipynb

run this file with:
> python module_3_duration_train_with_best_features.py --year=2025 --month=01
"""

import os

import pandas as pd
import xgboost as xgb
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

# MLFLOW_TRACKING_URI = "http://ec2-3-123-1-12.eu-central-1.compute.amazonaws.com:5000"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("module_3_nyc-taxi-experiment")

models_folder = 'models'
os.makedirs(models_folder, exist_ok=True)

DATA_FOLDER = '../data/'
DATA_FILE_FORMAT = "{}green_tripdata_{}-{:02d}.parquet"

def read_dataframe(year, month):
    df = pd.read_parquet(DATA_FILE_FORMAT.format(DATA_FOLDER, year, month))

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    return df

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.095853,
            'max_depth': 30,
            'min_child_weight': 1.060597,
            'objectvie': 'reg:linear',
            'reg_alpha': 0.0180602,
            'reg_lambda': 0.1165873,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
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


def run(year, month):
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1

    df_train = read_dataframe(year, month)
    df_val = read_dataframe(next_year, next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost model for NYC Taxi duration prediction")
    parser.add_argument('--year', type=int, required=True, help='Year for training data')
    parser.add_argument('--month', type=int, required=True, help='Month for training data')
    args = parser.parse_args()

    # Train the model for January 2025 and validate on February 2025
    run(args.year, args.month)
