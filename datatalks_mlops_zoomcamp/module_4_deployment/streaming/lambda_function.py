import json
import base64
import boto3

client = boto3.client('kinesis')

OUTPUT_STREAM_NAME = 'data_talks_club-mlops-course-ride_prediction-results'

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    return 10.0

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

    for record in event['Records']:
        try:
            print(f"Processed Kinesis Event - EventID: {record['eventID']}")
            
            record_data = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            print(f"Record Data: {record_data}")
            
            record_data_json = json.loads(record_data)
            ride = record_data_json['ride']
            ride_id = record_data_json['ride_id']

            features = prepare_features(ride)
            print(f"Prepared Features: {features}")

            prediction = predict(features)

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

            # TODO: Do interesting work based on the new data
        except Exception as e:
            print(f"An error occurred {e}")
            raise e

    try:
        response = send_results(prediction_results)
    except Exception as e:
        print("Failed to send record to Kinesis", e)

    print(f"Successfully processed records and emitted {len(event['Records'])} results.")

    return {
        'prediction_results': prediction_results
    }
