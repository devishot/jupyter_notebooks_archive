import json
import sys

import requests
from deepdiff import DeepDiff

KINESIS_EVENT_FORMAT = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": None,  # to be set
                "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                "data": None,  # to be set
                "approximateArrivalTimestamp": 1654161514.132,
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::XXXXXXXXX:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:XXXXXXXXX:stream/ride_events",
        }
    ]
}

LAMBDA_URI = 'http://localhost:8080/2015-03-31/functions/function/invocations'

event = {
    "Data": "eyJyaWRlIjogeyJQVUxvY2F0aW9uSUQiOiAxMCwgIkRPTG9jYXRpb25JRCI6IDUwLCAidHJpcF9kaXN0YW5jZSI6IDQwfSwgInJpZGVfaWQiOiAiZTRlMWY5OTI0MmRiNDc3NmFiNTZkZmU2YTg3YTM5ZTgifQ==",
    "PartitionKey": "e4e1f99242db4776ab56dfe6a87a39e8",
}
expected_prediction = {
    'prediction_results': [
        {
            'model': 'ride_duration_prediction_model',
            'version': '8d6b2c8289b94d1cb40a313e0bf92aca',
            'prediction': {
                'ride_duration': 49.7,
                'ride_id': 'e4e1f99242db4776ab56dfe6a87a39e8',
            },
        }
    ]
}

partitionKey = event['PartitionKey']
data = event['Data']

event = KINESIS_EVENT_FORMAT.copy()
event['Records'][0]['kinesis']['partitionKey'] = partitionKey
event['Records'][0]['kinesis']['data'] = data

actual_response = requests.post(LAMBDA_URI, json=event).json()
# print(actual_response)

diff = DeepDiff(expected_prediction, actual_response, significant_digits=1)
print('Diff:', diff)

assert 'type_changes' not in diff
assert 'value_changes' not in diff
assert not diff, f"Differences found: {diff}"
print('Test passed')
