import sys
import json
import lambda_function

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

if __name__ == '__main__':
    events_file = sys.argv[1]  # 'events.json'

    with open(events_file) as f:
        events = json.load(f)

    for event in events:
        partitionKey = event['PartitionKey']
        data = event['Data']

        event = KINESIS_EVENT_FORMAT.copy()
        event['Records'][0]['kinesis']['partitionKey'] = partitionKey
        event['Records'][0]['kinesis']['data'] = data
        lambda_function.lambda_handler(event, None)
