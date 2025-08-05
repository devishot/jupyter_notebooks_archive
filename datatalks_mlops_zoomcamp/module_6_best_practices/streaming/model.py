import json
import base64
import boto3
import os

model = None

def load_model(model_uri: str):
    import mlflow
    global model
    if model is None:
        print(f'Loading model from uri: {model_uri}')
        model = mlflow.pyfunc.load_model(model_uri)
    return model


class KinesisCallback():
    def __init__(self, output_stream_name):
        self.output_stream_name = output_stream_name
        self.client = self.create_client()

    def create_client(self):
        endpoint_url = os.getenv('KINESIS_ENDPOINT_URL')
        if endpoint_url:
            self.client = boto3.client('kinesis', endpoint_url=endpoint_url)
        else:
            self.client = boto3.client('kinesis')

    def put_record(self, record):
        ride_id = record['prediction']['ride_id']
        self.client.put_record(
            StreamName=self.output_stream_name,
            Data=json.dumps(record),
            PartitionKey=ride_id
        )

class ModelService():
    def __init__(self, model, model_version, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def parse_record(self, record):
        record_data = base64.b64decode(record['kinesis']['data']).decode('utf-8')
        print(f"Record Data: {record_data}")
        
        record_data_json = json.loads(record_data)
        ride = record_data_json['ride']
        ride_id = record_data_json['ride_id']
        return ride, ride_id

    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self, X):
        y_pred = self.model.predict(X)
        return float(y_pred[0])

    def lambda_handler(self, event, context):
        # pylint: disable=unused-argument
        prediction_results = []

        for record in event['Records']:
            try:
                print(f"Processed Kinesis Event - EventID: {record['eventID']}")

                ride, ride_id = self.parse_record(record)
                print(f"Parsed Ride - RideID: {ride_id}, Ride: {ride}")

                features = self.prepare_features(ride)
                print(f"Prepared Features: {features}")

                prediction = self.predict(features)

                record_result = {
                    'model': 'ride_duration_prediction_model',
                    'version': self.model_version,
                    'prediction': {
                        'ride_duration': prediction,
                        'ride_id': ride_id
                    }
                }
                print(f"Prediction Result: {record_result}")

                for callback in self.callbacks:
                    callback(record_result)

                prediction_results.append(record_result)

            except Exception as e:
                print(f"An error occurred {e}")
                raise e

        return {
            'prediction_results': prediction_results
        }

def init(model_path: str, run_id: str, prediction_stream_name: str, test_run: bool):
    loaded_model = load_model(model_path)

    if test_run:
        return ModelService(loaded_model, run_id)

    stream = KinesisCallback(prediction_stream_name)
    return ModelService(loaded_model, run_id, callbacks=[stream.put_record])
