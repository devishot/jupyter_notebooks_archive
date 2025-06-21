import os
import json
import base64
import boto3

client = boto3.client('kinesis')

DRY_RUN = os.environ.get("DRY_RUN", False)

RUN_ID = os.environ.get("RUN_ID", '8d6b2c8289b94d1cb40a313e0bf92aca')
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME", 'data-talks-club-mlops-remote-s3-bucket')

OUTPUT_STREAM_NAME = os.environ.get("OUTPUT_STREAM_NAME", 'data_talks_club-mlops-course-ride_prediction-results')

model = None

def load_model():
    import mlflow
    global model
    if model is None:
        model_uri = f's3://{MODEL_BUCKET_NAME}/1/{RUN_ID}/artifacts/model'
        print(f'Loading model from uri: {model_uri}')
        model = mlflow.pyfunc.load_model(model_uri)
    return model

def parse_record(record):
    record_data = base64.b64decode(record['kinesis']['data']).decode('utf-8')
    print(f"Record Data: {record_data}")
    
    record_data_json = json.loads(record_data)
    ride = record_data_json['ride']
    ride_id = record_data_json['ride_id']
    return ride, ride_id

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(model, X):
     y_pred = model.predict(X)
     return float(y_pred[0])

def send_results(results):
    output_records = []
    for result in results:
        output_records.append({
            'Data': json.dumps(result),
            'PartitionKey': result['prediction']['ride_id']
        })

    client.put_records(
        StreamName=OUTPUT_STREAM_NAME,
        Records=output_records
    )

def lambda_handler(event, context):
    prediction_results = []

    try:
        loaded_model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

    for record in event['Records']:
        try:
            print(f"Processed Kinesis Event - EventID: {record['eventID']}")

            ride, ride_id = parse_record(record)
            print(f"Parsed Ride - RideID: {ride_id}, Ride: {ride}")

            features = prepare_features(ride)
            print(f"Prepared Features: {features}")

            prediction = predict(model, features)

            record_result = {
                'model': 'ride_duration_prediction_model',
                'version': '123',
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id
                }
            }
            print(f"Prediction Result: {record_result}")

            prediction_results.append(record_result)

        except Exception as e:
            print(f"An error occurred {e}")
            raise e

    if not DRY_RUN:
        try:
            send_results(prediction_results)
        except Exception as e:
            print("Failed to send record to Kinesis", e)

        print(f"Successfully processed records and emitted {len(event['Records'])} results.")

    return {
        'prediction_results': prediction_results
    }
