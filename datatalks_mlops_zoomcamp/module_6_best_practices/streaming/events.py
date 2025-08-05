import json
import base64
import uuid

EVENTS = [
    {
        "ride": {"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40},
        "ride_id": None,
    }
]

records = []
for event in EVENTS:
    ride_id = uuid.uuid4().hex
    event['ride_id'] = ride_id
    records.append(
        {
            "Data": base64.b64encode(json.dumps(event).encode("utf-8")).decode("utf-8"),
            "PartitionKey": ride_id,
        }
    )

with open("events.json", "w") as f:
    json.dump(records, f)
