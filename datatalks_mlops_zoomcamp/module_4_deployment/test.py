import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 40
}

URI = 'http://localhost:9696'

response = requests.post(f'{URI}/predict', json=ride)
print(response.json())
