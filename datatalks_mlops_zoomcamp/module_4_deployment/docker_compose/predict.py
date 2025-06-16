import os

import pickle

from flask import Flask, request, jsonify

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
RUN_ID = os.environ.get("RUN_ID")

# print(f'setting MLFLOW_TRACKING_URI to {MLFLOW_TRACKING_URI}')
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# model_uri = f'runs:/{RUN_ID}/model'
model_uri = f's3://bucket/1/{RUN_ID}/artifacts/model'
print(f'Loading model using uri: {model_uri}')
model = mlflow.pyfunc.load_model(model_uri)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(X):
    y_pred = model.predict(X)
    return y_pred

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)
    result = {
        'duration': pred[0]
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)