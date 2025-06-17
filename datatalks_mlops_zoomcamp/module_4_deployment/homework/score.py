#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd
import numpy as np

categorical = ['PULocationID', 'DOLocationID']

def read_data(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def prepare_results(df, y_pred):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    return df_result

def run():
    model_path = sys.argv[1] # '../../module_1/models/lin_reg.bin'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3

    df = read_data(year, month)
    dv, model = load_model(model_path)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # # Compute standard deviation
    # std_dev = np.std(y_pred)
    # print(f"Standard deviation of predicted durations: {std_dev:.2f}")

    # mean predicted duration
    mean_pred = np.mean(y_pred)
    print(f"Mean predicted duration: {mean_pred:.2f}")

    df_result = prepare_results(df, y_pred)

    os.makedirs('output', exist_ok=True)

    output_file = f'output/yellow-prediction_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    run()
