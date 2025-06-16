import os

import pickle

from flask import Flask, request, jsonify

import mlflow
from mlflow import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
RUN_ID = os.environ.get("RUN_ID")

print(f'setting MLFLOW_TRACKING_URI to {MLFLOW_TRACKING_URI}')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

client = MlflowClient()

# List artifacts at the base path
artifacts = client.list_artifacts(RUN_ID, path='preprocessor')
print("Artifacts under preprocessor: ", artifacts)


path = client.download_artifacts(run_id=RUN_ID, path='preprocessor/preprocessor.b')
print(f'Downloading DictVectorizer from path: {path}')
with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

model_uri = f'runs:/{RUN_ID}/model'
print(f'Loading model using id: {model_uri}')
model = mlflow.pyfunc.load_model(model_uri)

## Load the model from disk
# with open(f'models/lin_reg.bin', "rb") as f_in:
#     dv, model = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(X):    
    X = dv.transform(X)
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